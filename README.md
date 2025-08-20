# NotebookLM Server

A fast, optimized backend server for building personal knowledge bases with AI-powered chat capabilities.

## 🚀 Features

- **Vector Database Integration**: Uses Qdrant for efficient similarity search
- **Multiple Content Types**: Support for PDF, text, and web content
- **AI Chat**: GPT-4 powered responses based on your knowledge base
- **Optimized Performance**: Streamlined operations for faster response times
- **RESTful API**: Clean, intuitive endpoints for easy integration

## 🛠️ Prerequisites

- Node.js 18+ 
- Qdrant Vector Database
- OpenAI API Key

## 📦 Installation

1. **Clone the repository**
   ```bash
   cd server
   npm install
   ```

2. **Set up environment variables**
   Create a `.env` file in the server directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   QDRANT_URL=http://localhost:6333
   ```

3. **Start Qdrant Database**
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or use the provided docker-compose
   docker-compose up -d
   ```

4. **Start the server**
   ```bash
   npm start
   # or
   node index.js
   ```

The server will run on `http://localhost:3001`

## 🔌 API Endpoints

### Health Check
- `GET /test` - Server status check

### Sources Management
- `GET /api/sources-context` - Get all sources
- `DELETE /api/sources/clear` - Clear all sources
- `DELETE /api/sources/:id` - Remove specific source

### Content Upload
- `POST /api/embed-pdf` - Upload PDF file
- `POST /api/embed-text` - Add text content
- `POST /api/embed-url` - Add web content

### Chat
- `POST /api/chat` - Generate AI response

## 📊 Vector Database

- **Model**: OpenAI text-embedding-3-large (3072 dimensions)
- **Distance Metric**: Cosine similarity
- **Collection**: `personal-notebooklm`
- **Retrieval**: Top 5 most relevant chunks per query

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `QDRANT_API_KEY`: Qdrant API key (optional)
- `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333)
- `DOCKER_URL`: Alternative Qdrant URL for Docker environments

### Performance Settings
- Vector size: 3072 dimensions
- Retrieval limit: 5 chunks per query
- Collection management: Smart creation/deletion

## 📁 Project Structure

```
server/
├── index.js          # Main server file
├── package.json      # Dependencies
├── docker-compose.yml # Qdrant setup
├── uploads/          # Temporary file storage
└── README.md         # This file
```

## 🚀 Performance Optimizations

- **Smart Collection Management**: No unnecessary recreations
- **Consolidated Operations**: Reduced API calls
- **Efficient Error Handling**: Fast error responses
- **Minimal Logging**: Production-ready performance

## 🔒 Security Features

- CORS enabled for cross-origin requests
- File upload validation (PDF only)
- Input sanitization
- Environment variable protection

## 🐛 Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   - Ensure Qdrant is running on the specified port
   - Check firewall settings
   - Verify URL in environment variables

2. **OpenAI API Errors**
   - Verify API key is correct
   - Check API quota and rate limits
   - Ensure model access permissions

3. **File Upload Issues**
   - Check file size limits
   - Verify file format (PDF only)
   - Ensure uploads directory has write permissions

### Debug Mode
Enable verbose logging by setting `NODE_ENV=development`

## 📈 Monitoring

The server provides basic health endpoints for monitoring:
- `/test` - Basic connectivity
- `/api/sources-context` - Source count and status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
