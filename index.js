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

// Clear context on server restart (for development/testing)
console.log("Server starting - clearing previous context");
globalSourcesContext = [];

// Check if we're in a serverless environment (like Vercel)
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
  origin: true, // Allow all origins for testing
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

  // Use environment variable with fallback
  const qdrantUrl = (process.env.DOCKER_URL || process.env.QDRANT_URL || "http://localhost:6333").replace(':6333', '');
  console.log("Using Qdrant URL:", qdrantUrl);

  try {
    // Test connection to Qdrant first
    const collections = await qdrantClient.getCollections();
    console.log("Qdrant connection test successful, collections:", collections.collections.length);

    await QdrantVectorStore.fromDocuments(docs, embeddings, {
      url: qdrantUrl,
      collectionName: "personal-notebooklm",
      apiKey: process.env.QDRANT_API_KEY,
    });
    console.log("Documents stored in Qdrant");

    // Update global context with new source
    updateGlobalContext(metadata, text);
  } catch (error) {
    console.error("Qdrant connection error:", error);
    if (error.message.includes("fetch failed") || error.message.includes("ENOTFOUND")) {
      throw new Error(`Cannot connect to Qdrant at ${qdrantUrl}. Please check if the database is running and accessible.`);
    }
    throw error;
  }
}

// Function to update global context when new sources are added
function updateGlobalContext(metadata, content) {
  const sourceInfo = {
    id: `source_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
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
  console.log("Added source:", sourceInfo.title, "with ID:", sourceInfo.id);
}

// Function to generate dynamic system prompt
function generateSystemPrompt(relevantChunks) {
  console.log("Relevant Chunks: ", relevantChunks);
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

    const qdrantUrl = (process.env.DOCKER_URL || process.env.QDRANT_URL || "http://localhost:6333").replace(':6333', '');
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: qdrantUrl,
        collectionName: 'personal-notebooklm',
        apiKey: process.env.QDRANT_API_KEY,
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

    console.log("Response: ", response.choices[0].message.content);
    

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
    const qdrantUrl = (process.env.DOCKER_URL || process.env.QDRANT_URL || "http://localhost:6333").replace(':6333', '');
    const vectorStore = await QdrantVectorStore.fromExistingIndex(embeddings, {
      url: qdrantUrl,
      collectionName: "personal-notebooklm",
      apiKey: process.env.QDRANT_API_KEY,
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

// Test Qdrant connection
app.get("/test-qdrant", async (req, res) => {
  try {
    const qdrantUrl = (process.env.DOCKER_URL || process.env.QDRANT_URL || "http://localhost:6333").replace(':6333', '');
    console.log("Testing Qdrant connection to:", qdrantUrl);
    
    const collections = await qdrantClient.getCollections();
    
    res.json({ 
      message: "Qdrant connection successful",
      url: qdrantUrl,
      collections: collections.collections,
      totalCollections: collections.collections.length
    });
  } catch (error) {
    console.error("Qdrant connection test error:", error);
    res.status(500).json({ 
      error: "Qdrant connection test failed",
      message: error.message,
      url: (process.env.DOCKER_URL || process.env.QDRANT_URL || "http://localhost:6333").replace(':6333', '')
    });
  }
});

// Get current sources context
app.get("/api/sources-context", (req, res) => {
  res.json({
    totalSources: globalSourcesContext.length,
    sources: globalSourcesContext
  });
});

// Clear all sources and embeddings
app.delete("/api/sources/clear", async (req, res) => {
  try {
    console.log("Clearing all sources and embeddings...");
    
    // Clear global context
    globalSourcesContext = [];
    
    // Clear Qdrant collection
    const qdrantUrl = (process.env.DOCKER_URL || process.env.QDRANT_URL || "http://localhost:6333").replace(':6333', '');
    
    try {
      // Delete the entire collection
      await qdrantClient.deleteCollection("personal-notebooklm");
      console.log("Qdrant collection deleted");
      
      // Recreate empty collection
      await qdrantClient.createCollection("personal-notebooklm", {
        vectors: {
          size: 1536, // OpenAI embedding size
          distance: "Cosine"
        }
      });
      console.log("New empty collection created");
    } catch (error) {
      console.log("Collection might not exist yet, continuing...");
    }
    
    res.json({ 
      message: "All sources and embeddings cleared successfully",
      totalSources: 0
    });
  } catch (err) {
    console.error("Error clearing sources:", err);
    res.status(500).json({ error: err.message });
  }
});

// Remove specific source
app.delete("/api/sources/:id", async (req, res) => {
  try {
    const { id } = req.params;
    console.log("Removing source with ID:", id);
    
    // Find and remove from global context
    const sourceIndex = globalSourcesContext.findIndex(source => source.id === id);
    if (sourceIndex === -1) {
      return res.status(404).json({ error: "Source not found" });
    }
    
    const removedSource = globalSourcesContext.splice(sourceIndex, 1)[0];
    console.log("Removed source:", removedSource.title);
    
    // Note: For now, we're not removing individual embeddings from Qdrant
    // as it's complex to identify which vectors belong to which source
    // The clear all endpoint handles complete cleanup
    
    res.json({ 
      message: "Source removed successfully",
      removedSource,
      totalSources: globalSourcesContext.length
    });
  } catch (err) {
    console.error("Error removing source:", err);
    res.status(500).json({ error: err.message });
  }
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

    const qdrantUrl = (process.env.DOCKER_URL || process.env.QDRANT_URL || "http://localhost:6333").replace(':6333', '');
    await QdrantVectorStore.fromDocuments(docs, embeddings, {
      url: qdrantUrl,
      collectionName: "personal-notebooklm",
      apiKey: process.env.QDRANT_API_KEY,
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

    // console.log("Chat message:", message);

    // Check if OpenAI API key is set
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "OPENAI_API_KEY environment variable is not set" });
    }

    const response = await chat(message);
    // console.log(response);
    

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
