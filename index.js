import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import OpenAI from 'openai';
import cors from "cors";
import 'dotenv/config';

const app = express();
const client = new OpenAI();

// Global variable to store all sources context
let globalSourcesContext = [];

const uploadsDir = path.join(process.cwd(), "uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

const upload = multer({ dest: uploadsDir }); 

app.use(cors({
  origin: [process.env.CLIENT_URL, "http://localhost:8080"], 
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept'],
  credentials: true,
  preflightContinue: false,
  optionsSuccessStatus: 204
}));

// Parse JSON and URL-encoded data
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Helper function to create embeddings and store in Qdrant
async function createAndStoreEmbeddings(text, metadata) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY environment variable is not set");
  }

  const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-large" });
  console.log("Embeddings created");

  // Create a simple document from text
  const docs = [{
    pageContent: text,
    metadata: metadata
  }];

  await QdrantVectorStore.fromDocuments(docs, embeddings, {
    url: "http://localhost:6333",
    collectionName: "personal-notebooklm",
  });
  console.log("Documents stored in Qdrant");

  // Update global context with new source
  updateGlobalContext(metadata, text);
}

// Function to update global context when new sources are added
function updateGlobalContext(metadata, content) {
  const sourceInfo = {
    type: metadata.type,
    title: metadata.title,
    content: content.substring(0, 200) + "...", // First 200 chars
    timestamp: metadata.timestamp || new Date().toISOString()
  };

  if (metadata.url) {
    sourceInfo.url = metadata.url;
  }

  globalSourcesContext.push(sourceInfo);
  console.log("Global context updated. Total sources:", globalSourcesContext.length);
}

// Function to generate dynamic system prompt
function generateSystemPrompt(relevantChunks) {
  const sourcesSummary = globalSourcesContext.map(source => 
    `- ${source.type.toUpperCase()}: ${source.title} (${source.timestamp})`
  ).join('\n');

  return `
    You are an AI assistant who helps resolve user queries based on the context available from their personal knowledge base.

    AVAILABLE SOURCES IN KNOWLEDGE BASE:
    ${sourcesSummary}

    RELEVANT CONTEXT FOR THIS QUERY:
    ${JSON.stringify(relevantChunks, null, 2)}

    INSTRUCTIONS:
    1. Only answer based on the available context from the sources above
    2. If the context doesn't contain enough information, say so
    3. Reference specific sources when possible
    4. Be helpful and accurate based on the provided information
    5. If asked about sources not in the context, inform the user
  `;
}

// Chat function integrated into the main file
async function chat(userInput) {
  try {
    const userQuery = userInput;

    // Ready the client OpenAI Embedding Model
    const embeddings = new OpenAIEmbeddings({
      model: 'text-embedding-3-large',
    });

    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: 'http://localhost:6333',
        collectionName: 'personal-notebooklm',
      }
    );

    const vectorSearcher = vectorStore.asRetriever({
      k: 5, // Increased to get more context
    });

    const relevantChunk = await vectorSearcher.invoke(userQuery);

    const SYSTEM_PROMPT = generateSystemPrompt(relevantChunk);

    const response = await client.chat.completions.create({
      model: 'gpt-4',
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: userQuery },
      ],
    });

    // Return the response in the expected format
    return {
      text: response.choices[0].message.content,
      sourceDocuments: relevantChunk
    };
  } catch (error) {
    console.error('Error in chat function:', error);
    throw error;
  }
}

// Test endpoint
app.get("/test", (req, res) => {
  res.json({ message: "Server is working!" });
});

// Test chat endpoint
app.get("/test-chat", async (req, res) => {
  try {
    console.log("Testing chat functionality...");
    
    // Check if OpenAI API key is set
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "OPENAI_API_KEY environment variable is not set" });
    }

    // Initialize embeddings and vector store
    const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-large" });
    const vectorStore = await QdrantVectorStore.fromExistingIndex(embeddings, {
      url: "http://localhost:6333",
      collectionName: "personal-notebooklm",
    });

    // Check if there are any documents in the collection
    const collectionInfo = await vectorStore.client.getCollection("personal-notebooklm");
    console.log("Collection info:", collectionInfo);

    res.json({ 
      message: "Chat test completed",
      collectionExists: true,
      collectionInfo: collectionInfo,
      currentSources: globalSourcesContext
    });

  } catch (err) {
    console.error("Error in chat test:", err);
    res.status(500).json({ error: err.message });
  }
});

// Get current sources context
app.get("/api/sources-context", (req, res) => {
  res.json({
    totalSources: globalSourcesContext.length,
    sources: globalSourcesContext
  });
});

app.post("/api/embed-pdf", upload.single("pdf"), async (req, res) => {
  console.log("PDF upload request received");
  
  try {
    if (!req.file) {
      console.log("No file in request");
      return res.status(400).json({ error: "No PDF uploaded" });
    }

    console.log("File received:", req.file.originalname);

    // Load PDF
    const loader = new PDFLoader(req.file.path);
    const docs = await loader.load();
    console.log("PDF loaded, pages:", docs.length);

    if (!process.env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY environment variable is not set");
    }

    const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-large" });
    console.log("Embeddings created");

    await QdrantVectorStore.fromDocuments(docs, embeddings, {
      url: "http://localhost:6333", // your Qdrant instance
      collectionName: "personal-notebooklm",
    });
    console.log("Documents stored in Qdrant");

    // Update global context with PDF content
    const pdfContent = docs.map(doc => doc.pageContent).join("\n\n");
    updateGlobalContext({
      type: "pdf",
      title: req.file.originalname,
      timestamp: new Date().toISOString()
    }, pdfContent);

    fs.unlinkSync(req.file.path);
    console.log("Temp file deleted");

    res.status(200).json({ message: "PDF embedded successfully!" });
  } catch (err) {
    console.error("Error processing PDF:", err);
    res.status(500).json({ error: err.message });
  }
});

// New endpoint for text sources
app.post("/api/embed-text", async (req, res) => {
  console.log("Text embedding request received");
  
  try {
    const { title, content } = req.body;
    
    if (!title || !content) {
      return res.status(400).json({ error: "Title and content are required" });
    }

    console.log("Text received:", title, "Length:", content.length);

    // Create embeddings and store in Qdrant
    await createAndStoreEmbeddings(content, {
      type: "text",
      title: title,
      source: "user_input",
      timestamp: new Date().toISOString()
    });

    res.status(200).json({ message: "Text embedded successfully!" });
  } catch (err) {
    console.error("Error processing text:", err);
    res.status(500).json({ error: err.message });
  }
});

// New endpoint for URL sources
app.post("/api/embed-url", async (req, res) => {
  console.log("URL embedding request received");
  
  try {
    const { title, url } = req.body;
    
    if (!title || !url) {
      return res.status(400).json({ error: "Title and URL are required" });
    }

    console.log("URL received:", title, url);

    // Load content from URL
    const loader = new CheerioWebBaseLoader(url);
    const docs = await loader.load();
    console.log("URL content loaded, documents:", docs.length);

    if (docs.length === 0) {
      throw new Error("Could not extract content from URL");
    }

    // Combine all content from the URL
    const combinedContent = docs.map(doc => doc.pageContent).join("\n\n");
    
    // Create embeddings and store in Qdrant
    await createAndStoreEmbeddings(combinedContent, {
      type: "url",
      title: title,
      url: url,
      source: "web_scraping",
      timestamp: new Date().toISOString()
    });

    res.status(200).json({ message: "URL embedded successfully!" });
  } catch (err) {
    console.error("Error processing URL:", err);
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/chat", async (req, res) => {
  console.log("Chat request received");
  
  try {
    const { message } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    console.log("Chat message:", message);

    // Check if OpenAI API key is set
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "OPENAI_API_KEY environment variable is not set" });
    }

    const response = await chat(message);

    if (!response || !response.text) {
      return res.status(500).json({ error: "Failed to generate chat response" });
    }

    console.log("Chat response generated successfully");

    res.status(200).json({ 
      message: "Chat response generated successfully!",
      response: response.text,
      sources: response.sourceDocuments || []
    });

  } catch (err) {
    console.error("Error in chat:", err);
    
    // Handle specific error types with appropriate status codes
    if (err.message.includes("OPENAI_API_KEY")) {
      return res.status(500).json({ error: "OpenAI API key not configured" });
    }
    
    if (err.message.includes("Qdrant") || err.message.includes("vector")) {
      return res.status(503).json({ error: "Vector database connection failed" });
    }
    
    if (err.message.includes("rate limit") || err.message.includes("quota")) {
      return res.status(429).json({ error: "OpenAI API rate limit exceeded" });
    }
    
    // Default error response
    res.status(500).json({ error: err.message || "Internal server error" });
  }
});

app.listen(3001, () => console.log("Backend running on http://localhost:3001"));
