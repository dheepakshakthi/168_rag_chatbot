# New Features - Chat History & File Management

This document describes the new features added to the 168 RAG Chatbot.

## 🗂️ Chat History Management

### Features
- **Persistent Chat Sessions**: All conversations are automatically saved locally
- **Session List**: View all your previous chat sessions in a sidebar
- **Session Preview**: See first message, date, time, and message count
- **Load Previous Chats**: Click any session to load the full conversation
- **New Chat**: Start a fresh conversation anytime
- **Delete Sessions**: Remove individual chat histories

### How to Use
1. Click the **"History"** button in the header to open the history sidebar
2. Browse through your previous conversations
3. Click on any session to load it
4. Click the **"New Chat"** button to start fresh
5. Hover over a session and click the trash icon to delete it

### Storage
- Chat histories are saved in `chat_history/` folder
- Each session is a JSON file with format: `session_TIMESTAMP.json`
- Contains all messages, roles, sources, and timestamps

### Structure
```json
{
    "session_id": "session_1234567890",
    "created_at": "2024-01-15T10:30:00",
    "updated_at": "2024-01-15T10:35:00",
    "messages": [
        {
            "role": "user",
            "content": "Question",
            "timestamp": "2024-01-15T10:30:00"
        },
        {
            "role": "assistant",
            "content": "Answer",
            "sources": [...],
            "timestamp": "2024-01-15T10:30:05"
        }
    ]
}
```

## 📁 File Management

### Features
- **Files List**: View all uploaded documents in a sidebar
- **File Details**: See filename, size, upload date, and file type icon
- **Individual File Deletion**: Remove specific files without clearing all
- **Auto-Refresh**: Files list updates automatically after upload/delete
- **File Icons**: Visual indicators for PDFs, HTML, code, and text files

### How to Use
1. Click the **"Files"** button in the header to open the files sidebar
2. Browse through your uploaded documents
3. Hover over a file and click the X icon to delete it
4. Upload new files and they'll appear immediately

### File Information
Each file shows:
- **Icon**: 📕 for PDF, 🌐 for HTML, 💻 for code, 📄 for text
- **Filename**: Full name with extension
- **Size**: In kilobytes (KB)
- **Date**: When the file was uploaded

### Smart Deletion
- Deleting individual files triggers automatic vectorstore rebuild
- If you delete the last file, the system switches to general mode
- Rebuild happens in the background - you'll see a notification

## 🎨 User Interface Updates

### Sidebar Design
- **Slide-in Animation**: Smooth transitions when opening/closing
- **Responsive Layout**: Adapts to mobile screens (slides from bottom)
- **Active Session Highlight**: Current chat is highlighted in blue
- **Hover Effects**: Visual feedback on all interactive elements

### Button Additions
- **History Button**: Toggle chat history sidebar
- **Files Button**: Toggle files list sidebar
- **New Chat Button**: Inside history sidebar footer

### Notifications
- Success messages when loading/deleting sessions
- Info messages for file operations with 7-second duration
- Error messages for failed operations

## 🔧 Backend API Endpoints

### Chat History Endpoints

#### GET `/history/sessions`
- Lists all chat sessions
- Returns session metadata (ID, preview, date, message count)

#### GET `/history/<session_id>`
- Retrieves full chat history for a session
- Returns all messages with roles, content, sources

#### DELETE `/history/<session_id>`
- Deletes a chat session
- Removes the JSON file from disk

### File Management Endpoints

#### GET `/files`
- Lists all uploaded files
- Returns filename, size, upload date

#### DELETE `/files/<filename>`
- Deletes a specific file
- Triggers vectorstore rebuild
- Switches to general mode if no files remain

## 📱 Responsive Design

### Desktop (>768px)
- Sidebars appear on left (history) and right (files)
- Fixed position, slide in from sides
- 300px width, full height

### Mobile (≤768px)
- Sidebars slide up from bottom
- Full width, 70% viewport height
- Rounded top corners
- Better touch targets

## 🚀 Usage Examples

### Example 1: Continuing a Previous Conversation
1. Open the app
2. Click "History" button
3. Find your session from yesterday
4. Click to load it
5. Continue asking questions

### Example 2: Managing Documents
1. Upload several PDF files
2. Click "Files" button to see them
3. Notice one file is outdated
4. Hover and click X to delete just that file
5. System rebuilds vectorstore automatically

### Example 3: Starting Fresh
1. Finish a conversation
2. Click "History" button
3. Click "New Chat" button
4. Ask a new question

## 🔐 Privacy & Data

- **Local Storage**: All data stays on your machine
- **No Cloud Sync**: Chat histories are not uploaded anywhere
- **Manual Cleanup**: Delete sessions whenever you want
- **File Control**: Remove documents at any time

## 🎯 Benefits

1. **Conversation Continuity**: Pick up where you left off
2. **Organization**: Multiple chat sessions for different topics
3. **Document Management**: Track what files are in your RAG system
4. **Flexibility**: Delete old chats and files to save space
5. **User Experience**: Intuitive sidebar navigation

## 🛠️ Technical Details

### Frontend (JavaScript)
- Session ID generation: `session_${Date.now()}`
- State management for sidebars
- Async fetch calls to backend
- DOM manipulation for dynamic content

### Backend (Python/Flask)
- JSON file-based storage
- Automatic folder creation
- File system operations with error handling
- Vectorstore management

### Styling (CSS)
- CSS Grid and Flexbox layouts
- CSS animations and transitions
- Custom scrollbar styling
- Media queries for responsive design

## 🐛 Known Limitations

1. **Large Files**: Very large chat histories may be slow to load
2. **No Search**: Can't search within chat history (yet)
3. **No Export**: Can't export chats to other formats (yet)
4. **Single User**: Designed for single-user local use

## 🔮 Future Enhancements (Potential)

- Search functionality within chat history
- Export chats to PDF/TXT
- Tag/categorize chat sessions
- File preview in sidebar
- Drag-and-drop file upload
- Bulk file operations

---

**Last Updated**: 2024-01-15
**Version**: 2.0.0
