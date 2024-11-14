import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { File } from "parse-diff";
import minimatch from "minimatch";
import Anthropic from "@anthropic-ai/sdk";
import { TextBlock } from "@anthropic-ai/sdk/resources";

const MAX_OUTPUT_TOKENS: number = Number(core.getInput("MAX_OUTPUT_TOKENS")) || 4096;
const MAX_CONTEXT_TOKENS: number = Number(core.getInput("MAX_CONTEXT_TOKENS")) || 16384;
const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const AI_MODEL: string = core.getInput("AI_MODEL");
const ANTHROPIC_API_KEY: string = core.getInput("ANTHROPIC_API_KEY");
const MODEL_PROVIDER: string = core.getInput("MODEL_PROVIDER") || "openai"; // 기본값은 openai
const IGNORE_PATTERNS: string = core.getInput("IGNORE_PATTERNS") || "";

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

const anthropic = new Anthropic({
  apiKey: ANTHROPIC_API_KEY,
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
  changedFiles?: string[];
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8")
  );

  // PR 기본 정보 가져오기
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });

  // 변경된 파일 목록 가져오기
  const filesResponse = await octokit.pulls.listFiles({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });

  const changedFiles = filesResponse.data.map(file => file.filename);

  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
    changedFiles, // 변경된 파일 목록 추가
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });
  // @ts-expect-error - response.data is a string
  return response.data;
}

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = [];

  for (const file of parsedDiff) {
    if (file.to === "/dev/null") continue; // Ignore deleted files
      const prompt = createPrompt(file, prDetails);
      console.log(prompt);
      const aiResponse = await getAIResponse(file, prompt);
      if (aiResponse) {
        const newComments = createComment(file, aiResponse);
        if (newComments) {
          comments.push(...newComments);
        }
      }
  }
  return comments;
}

function createPrompt(file: File, prDetails: PRDetails): string {
  return `

# Pull Request Details
### Pull Request Info
- title
  > ${prDetails.title}
- description
  > ${prDetails.description}

### Code for review
- file

  > ${file.to}

- multiple diffs
  data is type of parseDiff.File.chunks.
  \`\`\` json
  ${JSON.stringify(file.chunks)}
  \`\`\`

- the json data schema is look like this: Chunk[]
  \`\`\`
  export interface Chunk {
    content: string;
    changes: Change[];
    oldStart: number;
    oldLines: number;
    newStart: number;
    newLines: number;
  }

  export interface NormalChange {
    type: 'normal';
    ln1: number;
    ln2: number;
    normal: true;
    content: string;
  }

  export interface AddChange {
    type: 'add';
    add: true;
    ln: number;
    content: string;
  }

  export interface DeleteChange {
    type: 'del';
    del: true;
    ln: number;
    content: string;
  }

  export type ChangeType = 'normal' | 'add' | 'del';

  export type Change = NormalChange | AddChange | DeleteChange;
  \`\`\`


# Reviewing Guide

### Answer Role: Reviewer
당신은 개발팀을 이루는 팀장입니다. 팀원이 작성한 코드를 검토하는 역할을 수행합니다.

### Requirements, Reviewing Guide, Pull Request on GitHub
- 매 코드 리뷰잉은, Pull Request의 제목과 설명을 고려하세요.
- 극심한 코드스멜, 보안의 위험, 버그 발생가능성, 비효율적인 코드 만을 포함해주세요.
- GitHub Markdown 형식을 사용하여 댓글을 작성하세요.
- 코드와 관련된 댓글만 포함하세요.
- 리뷰잉의 근거를 명확하게 작성하세요.

### Don't include
- 검토 및 고려사항은 같은 모호한것들은 절대 포함하지 마세요.
- 권장의 늬앙스가 담긴 코드 리뷰는 하지 마세요.
- 코드에 대한 긍정적인 댓글이나 칭찬을 포함하지 마세요.
- 코드에 주석 추가를 절대 제안하지 마세요.

### Additional Requirements

${process.env.ADDITIONAL_REQUIREMENTS}

### Output Format
- 다음 JSON 형식으로 응답 메시지를 제공하세요: {"reviews": [{"lineNumber": "<line_number>", "reviewComment": "<review_comment>"}]}
    - <line_number>
      - <line_number> 는 코드 줄 번호를 의미합니다.
      - '### Code for review'에서 각 코드 줄 번호를 참고하세요.
      - 코드 줄 번호가 연속적이라면 범위로 표시해주세요. 예시: "1-3"
      - 코드에 줄 번호를 표시할 수 없다면 해당 리뷰는 제외해주세요.
    - <review_comment>
      - <review_comment>는 리뷰잉한 댓글을 의미합니다.
      - <review_comment>가 애매하다면 해당 리뷰는 제외해주세요.
- 해당 코드에 리뷰가 없다면 "reviews"를 빈 배열로 두세요.
- <review_comment> 의 내용은 한글로 출력해주세요.
`;

}

function estimateTokenCount(text: string): number {
  // 대략적인 추정: 영어 기준 1토큰 ≈ 4글자
  // 한글은 1글자당 약 2-3토큰 소비
  const englishCharCount = text.replace(/[^a-zA-Z0-9\s]/g, '').length;
  const koreanCharCount = text.replace(/[^ㄱ-ㅎㅏ-ㅣ가-힣]/g, '').length;

  return Math.ceil(englishCharCount / 4) + (koreanCharCount * 2);
}

async function getAIResponse(file: File, prompt: string): Promise<Array<{
  lineNumber: string;
  reviewComment: string;
}> | null> {
  try {
    const estimatedTokens = estimateTokenCount(prompt);
    console.log(`Estimated input token count: ${estimatedTokens}`);

    if (estimatedTokens > MAX_CONTEXT_TOKENS) {
      console.warn(`${file.to}: Input tokens (${estimatedTokens}) is too large`);
      return [];
    }

    if (MODEL_PROVIDER === "anthropic") {
      const response = await anthropic.messages.create({
        model: AI_MODEL,
        max_tokens: MAX_OUTPUT_TOKENS,
        temperature: 0.2,
        messages: [
          {
            role: "user",
            content: prompt,
          },
        ],
      });
      if (response.content.length === 0) {
        return [];
      }
      const textBlock = response.content[0] as TextBlock;
      const res = textBlock.text.trim() || '{}';
      const cleanedJsonString = res.replace(/```json|```/g, "");
      return JSON.parse(cleanedJsonString).reviews;
    } else {
      const queryConfig = {
        model: AI_MODEL,
        temperature: 0.2,
        max_tokens: MAX_OUTPUT_TOKENS,
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0,
      };

      const response = await openai.chat.completions.create({
        ...queryConfig,
        ...(AI_MODEL === "gpt-4-1106-preview"
          ? { response_format: { type: "json_object" } }
          : {}),
        messages: [
          {
            role: "system",
            content: prompt,
          },
        ],
      });
      if (response.choices.length === 0) {
        return [];
      }
      const res = response.choices[0].message?.content?.trim() || "{}";
      const cleanedJsonString = res.replace(/```json|```/g, "");
      return JSON.parse(cleanedJsonString).reviews;
    }
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

function createComment(
  file: File,
  aiResponses: Array<{
    lineNumber: string;
    reviewComment: string;
  }>
): Array<{ body: string; path: string; line: number; start_line?: number }> {
  return aiResponses.flatMap((aiResponse) => {
    if (!file.to) {
      return [];
    }
    if (!aiResponse.lineNumber || !aiResponse.reviewComment) {
      return [];
    }

    if (aiResponse.lineNumber.includes('-')) {
      const [start, end] = aiResponse.lineNumber.split('-').map(Number);
      return {
        body: aiResponse.reviewComment,
        path: file.to,
        start_line: start,
        line: end,
      };
    }

    return {
      body: aiResponse.reviewComment,
      path: file.to,
      line: Number(aiResponse.lineNumber),
    };
  });
}

async function createReviewComments(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number; start_line?: number }>
): Promise<void> {
  const commentsWithSide = comments.map(comment => ({
    ...comment,
    side: 'RIGHT',
    start_side: comment.start_line ? 'RIGHT' : undefined,
  }));

  await octokit.pulls.createReview({
    owner,
    repo,
    pull_number,
    comments: commentsWithSide,
    event: 'COMMENT',
  });
}

async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  );

  if (eventData.action === "opened") {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number
    );
  } else if (eventData.action === "synchronize") {
    const newBaseSha = eventData.before;
    const newHeadSha = eventData.after;
    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });
    response.data.files = response.data.files?.filter(f => prDetails.changedFiles?.indexOf(f.filename) !== -1) // exclude files that are not changed
    diff = String(response.data);
  } else {
    console.log("Unsupported event:", process.env.GITHUB_EVENT_NAME);
    return;
  }

  if (!diff) {
    console.log("No diff found");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = IGNORE_PATTERNS
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });

  const comments = await analyzeCode(filteredDiff, prDetails);
  if (comments.length > 0) {
    await createReviewComments(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments,
    );
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
