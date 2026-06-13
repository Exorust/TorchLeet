import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { questions } from "./data/questions.js";

import { listQuestionsSchema, listQuestions } from "./tools/list-questions.js";
import { getQuestionSchema, getQuestion } from "./tools/get-question.js";
import { getLearningPathSchema, getLearningPath } from "./tools/get-learning-path.js";
import { getHintSchema, getHint } from "./tools/get-hint.js";
import { checkSolutionSchema, checkSolution } from "./tools/check-solution.js";
import { getCompanyPrepSchema, getCompanyPrep } from "./tools/get-company-prep.js";
import { getPrerequisitesSchema, getPrerequisites } from "./tools/get-prerequisites.js";

import { TUTOR_PROMPT, getTutorMessages } from "./prompts/tutor.js";
import {
  INTERVIEW_PREP_PROMPT,
  interviewPrepSchema,
  getInterviewPrepMessages,
} from "./prompts/interview-prep.js";
import { REVIEW_PROMPT, getReviewMessages } from "./prompts/review.js";
import {
  EXPLAIN_PROMPT,
  explainSchema,
  getExplainMessages,
} from "./prompts/explain.js";

export function createServer(): McpServer {
  const server = new McpServer({
    name: "torchleet",
    version: "0.1.0",
  });

  // --- Tools ---

  server.tool(
    "list_questions",
    "Browse and filter TorchLeet questions by set, difficulty, company, category, or track",
    listQuestionsSchema,
    async (params) => listQuestions(params),
  );

  server.tool(
    "get_question",
    "Get full details for a specific question including description, companies, and Colab links",
    getQuestionSchema,
    async (params) => getQuestion(params),
  );

  server.tool(
    "get_learning_path",
    "Get a structured learning path: LLM path (build an LLM from scratch), basics, or advanced",
    getLearningPathSchema,
    async (params) => getLearningPath(params),
  );

  server.tool(
    "get_hint",
    "Get a progressive hint for a question (level 1=conceptual, 2=signatures, 3=step-by-step). Never reveals the full solution.",
    getHintSchema,
    async (params) => getHint(params),
  );

  server.tool(
    "check_solution",
    "Review submitted code against a question's requirements — returns a verification checklist, does NOT execute code",
    checkSolutionSchema,
    async (params) => checkSolution(params),
  );

  server.tool(
    "get_company_prep",
    "Get a prioritized study plan for a specific company's ML interviews",
    getCompanyPrepSchema,
    async (params) => getCompanyPrep(params),
  );

  server.tool(
    "get_prerequisites",
    "Find prerequisite questions to complete before tackling a harder problem",
    getPrerequisitesSchema,
    async (params) => getPrerequisites(params),
  );

  // --- Prompts ---

  server.prompt(
    TUTOR_PROMPT.name,
    TUTOR_PROMPT.description,
    () => getTutorMessages(),
  );

  server.prompt(
    INTERVIEW_PREP_PROMPT.name,
    INTERVIEW_PREP_PROMPT.description,
    interviewPrepSchema,
    (params) => getInterviewPrepMessages(params),
  );

  server.prompt(
    REVIEW_PROMPT.name,
    REVIEW_PROMPT.description,
    () => getReviewMessages(),
  );

  server.prompt(
    EXPLAIN_PROMPT.name,
    EXPLAIN_PROMPT.description,
    explainSchema,
    (params) => getExplainMessages(params),
  );

  // --- Resources ---

  server.resource(
    "question-catalog",
    "torchleet://questions",
    { description: "Full TorchLeet question catalog as JSON", mimeType: "application/json" },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          text: JSON.stringify(questions, null, 2),
          mimeType: "application/json",
        },
      ],
    }),
  );

  return server;
}
