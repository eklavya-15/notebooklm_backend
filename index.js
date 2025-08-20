import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import OpenAI from 'openai';
import { QdrantClient } from '@qdrant/js-client-rest';
import cors from "cors";
import 'dotenv/config';

const app = express();
const client = new OpenAI();

// Initialize Qdrant client
const qdrantClient = new QdrantClient({
  url: (process.env.DOCKER_URL || process.env.QDRANT_URL || "http://localhost:6333").replace(':6333', ''),
  apiKey: process.env.QDRANT_API_KEY,
});

// Global variable to store all sources context
let globalSourcesContext = [];

// Check if we're in a serverless environment
const isServerless = process.env.VERCEL || process.env.NODE_ENV === 'production';

let uploadsDir;
if (!isServerless) {
  uploadsDir = path.join(process.cwd(), "uploads");
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }
}

const upload = multer({ dest: uploadsDir || '/tmp' }); 

app.use(cors({
  origin: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept'],
  credentials: true,
  preflightContinue: false,
  optionsSuccessStatus: 204
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Optimized collection management
async function ensureCollection() {
  try {
    await qdrantClient.getCollection("personal-notebooklm");
  } catch {
    await qdrantClient.createCollection("personal-notebooklm", {
      vectors: { size: 3072, distance: "Cosine" }
    });
  }
}

// Optimized embedding storage
async function storeEmbeddings(text, metadata) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY environment variable is not set");
  }

  const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-large" });
  const docs = [{ pageContent: text, metadata }];

  const qdrantUrl = (process.env.DOCKER_URL || process.env.QDRANT_URL || "http://localhost:6333").replace(':6333', '');
  
  await QdrantVectorStore.fromDocuments(docs, embeddings, {
    url: qdrantUrl,
    collectionName: "personal-notebooklm",
    apiKey: process.env.QDRANT_API_KEY,
    collectionConfig: {
      vectors: { size: 3072, distance: "Cosine" }
    }
  });

  // Update global context
  const sourceInfo = {
    id: `source_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    type: metadata.type,
    title: metadata.title,
    content: text.substring(0, 200) + "...",
    timestamp: metadata.timestamp || new Date().toISOString()
  };

  if (metadata.url) sourceInfo.url = metadata.url;
  globalSourcesContext.push(sourceInfo);
}

// Optimized system prompt generation
function generateSystemPrompt(relevantChunks) {
  const sourcesSummary = globalSourcesContext.map(source => 
    `- ${source.type.toUpperCase()}: ${source.title}`
  ).join('\n');

  return `You are an AI assistant who helps resolve user queries based on the context available from their personal knowledge base.

AVAILABLE SOURCES IN KNOWLEDGE BASE:
${sourcesSummary}

RELEVANT CONTEXT FOR THIS QUERY:
${JSON.stringify(relevantChunks, null, 2)}

INSTRUCTIONS:
1. Only answer based on the available context from the sources above
2. If the context doesn't contain enough information, say so
3. Reference specific sources when possible
4. Be helpful and accurate based on the provided information
5. If asked about sources not in the context, inform the user`;
}

// Optimized chat function
async function chat(userInput) {
  const embeddings = new OpenAIEmbeddings({ model: 'text-embedding-3-large' });
  const qdrantUrl = (process.env.DOCKER_URL || process.env.QDRANT_URL || "http://localhost:6333").replace(':6333', '');
  
  const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
    url: qdrantUrl,
    collectionName: 'personal-notebooklm',
    apiKey: process.env.QDRANT_API_KEY,
  });

  const relevantChunk = await vectorStore.asRetriever({ k: 5 }).invoke(userInput);
  const systemPrompt = generateSystemPrompt(relevantChunk);

  const response = await client.chat.completions.create({
    model: 'gpt-4',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userInput },
    ],
  });

  return {
    text: response.choices[0].message.content,
    sourceDocuments: relevantChunk
  };
}

// API Endpoints
app.get("/test", (req, res) => res.json({ message: "Server is working!" }));

app.get("/api/sources-context", (req, res) => {
  res.json({
    totalSources: globalSourcesContext.length,
    sources: globalSourcesContext
  });
});

app.delete("/api/sources/clear", async (req, res) => {
  try {
    globalSourcesContext = [];
    await qdrantClient.deleteCollection("personal-notebooklm");
    await ensureCollection();
    res.json({ message: "All sources cleared successfully", totalSources: 0 });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete("/api/sources/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const sourceIndex = globalSourcesContext.findIndex(source => source.id === id);
    if (sourceIndex === -1) return res.status(404).json({ error: "Source not found" });
    
    globalSourcesContext.splice(sourceIndex, 1)[0];
    res.json({ message: "Source removed successfully", totalSources: globalSourcesContext.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/embed-pdf", upload.single("pdf"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No PDF uploaded" });

    const loader = new PDFLoader(req.file.path);
    const docs = await loader.load();
    const pdfContent = docs.map(doc => doc.pageContent).join("\n\n");
    
    await storeEmbeddings(pdfContent, {
      type: "pdf",
      title: req.file.originalname,
      timestamp: new Date().toISOString()
    });

    fs.unlinkSync(req.file.path);
    res.status(200).json({ message: "PDF embedded successfully!" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/embed-text", async (req, res) => {
  try {
    const { title, content } = req.body;
    if (!title || !content) return res.status(400).json({ error: "Title and content are required" });

    await storeEmbeddings(content, {
      type: "text",
      title: title,
      timestamp: new Date().toISOString()
    });

    res.status(200).json({ message: "Text embedded successfully!" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/embed-url", async (req, res) => {
  try {
    const { title, url } = req.body;
    if (!title || !url) return res.status(400).json({ error: "Title and URL are required" });

    const loader = new CheerioWebBaseLoader(url);
    const docs = await loader.load();
    
    if (docs.length === 0) throw new Error("Could not extract content from URL");
    
    const combinedContent = docs.map(doc => doc.pageContent).join("\n\n");
    
    await storeEmbeddings(combinedContent, {
      type: "url",
      title: title,
      url: url,
      timestamp: new Date().toISOString()
    });

    res.status(200).json({ message: "URL embedded successfully!" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: "Message is required" });
    if (!process.env.OPENAI_API_KEY) return res.status(500).json({ error: "OpenAI API key not configured" });

    const response = await chat(message);
    
    if (!response || !response.text) {
      return res.status(500).json({ error: "Failed to generate chat response" });
    }

    res.status(200).json({ 
      message: "Chat response generated successfully!",
      response: response.text,
      sources: response.sourceDocuments || []
    });

  } catch (err) {
    if (err.message.includes("Qdrant") || err.message.includes("vector")) {
      return res.status(503).json({ error: "Vector database connection failed" });
    }
    if (err.message.includes("rate limit") || err.message.includes("quota")) {
      return res.status(429).json({ error: "OpenAI API rate limit exceeded" });
    }
    res.status(500).json({ error: err.message || "Internal server error" });
  }
});

app.listen(3001, () => console.log("Backend running on http://localhost:3001"));
