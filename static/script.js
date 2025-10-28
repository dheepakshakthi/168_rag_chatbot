// Auto-resize textarea
const messageInput = document.getElementById('messageInput');
const chatContainer = document.getElementById('chatContainer');
const sendBtn = document.getElementById('sendBtn');

messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Handle file upload
async function handleFileUpload(event) {
    const files = event.target.files;
    if (files.length === 0) return;
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }
    
    // Show upload status
    showNotification('Uploading and processing files...', 'info');
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification(data.message + ' - RAG mode enabled!', 'success');
            // Update UI
            window.hasDocuments = data.has_documents;
            updateHeaderMode(data.has_documents);
            // Enable clear docs button
            document.querySelector('.clear-docs-btn').disabled = false;
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showNotification('Upload failed: ' + error.message, 'error');
    }
    
    // Reset file input
    event.target.value = '';
}

// Clear documents
async function clearDocuments() {
    if (!confirm('Are you sure you want to delete all uploaded documents? This will also clear the vector database.')) {
        return;
    }
    
    showNotification('Clearing documents...', 'info');
    
    try {
        const response = await fetch('/clear_documents', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.needs_restart) {
                showNotification(data.message, 'info', 10000);
            } else {
                showNotification(data.message + ' - Switched to general mode', 'success');
                window.hasDocuments = false;
                updateHeaderMode(false);
                // Disable clear docs button
                document.querySelector('.clear-docs-btn').disabled = true;
            }
        } else {
            showNotification('Error: ' + data.error, 'error', 10000);
        }
    } catch (error) {
        showNotification('Failed to clear documents: ' + error.message, 'error');
    }
}

// Update header mode indicator
function updateHeaderMode(hasDocuments) {
    const modeText = document.querySelector('.logo-text p');
    if (modeText) {
        modeText.textContent = hasDocuments ? 'Your AI Assistant (RAG Mode)' : 'Your AI Assistant (General Mode)';
    }
}

// Show notification
function showNotification(message, type = 'info', duration = 5000) {
    // Remove existing notification
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Auto-remove after specified duration
    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

// Handle Enter key
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // Disable input while processing
    messageInput.disabled = true;
    sendBtn.disabled = true;
    
    // Remove welcome message if present
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    // Add user message
    addMessage(message, 'user');
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        // Send to backend
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (data.success) {
            // Add assistant response
            addMessage(data.response, 'assistant', data.sources);
        } else {
            // Show error
            addMessage('❌ Error: ' + data.error, 'assistant');
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage('❌ Connection error. Please make sure the server is running.', 'assistant');
    }
    
    // Re-enable input
    messageInput.disabled = false;
    sendBtn.disabled = false;
    messageInput.focus();
}

// Add message to chat
function addMessage(text, sender, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Format text (basic markdown-like formatting)
    const formattedText = formatText(text);
    contentDiv.innerHTML = formattedText;
    
    messageDiv.appendChild(contentDiv);
    
    // Add sources if available
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        
        let sourcesHTML = '<div class="sources-title">📚 Sources:</div>';
        sources.forEach((source, index) => {
            const relevance = source.page !== 'N/A' 
                ? `Page ${source.page} - Relevance: ${(1 - source.score).toFixed(2)}`
                : `Relevance: ${(1 - source.score).toFixed(2)}`;
            sourcesHTML += `<div class="source-item">
                ${index + 1}. ${source.source} (${relevance})
            </div>`;
        });
        
        sourcesDiv.innerHTML = sourcesHTML;
        contentDiv.appendChild(sourcesDiv);
    }
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Format text with basic markdown
function formatText(text) {
    // Escape HTML
    text = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    // Code blocks
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, function(match, lang, code) {
        return `<pre><code>${code.trim()}</code></pre>`;
    });
    
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

// Show typing indicator
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typing-' + Date.now();
    
    const indicatorDiv = document.createElement('div');
    indicatorDiv.className = 'typing-indicator';
    indicatorDiv.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    
    typingDiv.appendChild(indicatorDiv);
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return typingDiv.id;
}

// Remove typing indicator
function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

// Clear chat
async function clearChat() {
    if (confirm('Are you sure you want to clear the chat?')) {
        const ragStatus = window.hasDocuments 
            ? '<p class="rag-status">🟢 RAG Mode Active - I\'ll answer based on your uploaded documents</p>'
            : '<p class="rag-status">🔵 General Mode - Upload documents to enable RAG capabilities</p>';
        
        // Clear chat container
        chatContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">💡</div>
                <h2>Welcome to 168 AI Assistant!</h2>
                ${ragStatus}
                <p>I'm your AI assistant that can help you with various tasks:</p>
                <div class="feature-grid">
                    <div class="feature-item">
                        <span class="feature-icon">�</span>
                        <span>PDF Documents</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">🌐</span>
                        <span>HTML Files</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">�</span>
                        <span>Code Files</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">�</span>
                        <span>General Chat</span>
                    </div>
                </div>
                <p class="welcome-subtext">Upload your documents to get started with RAG, or just chat with me!</p>
                <div class="supported-formats">
                    <strong>Supported formats:</strong> PDF, HTML, Python, Java, JavaScript, C/C++, C#, Ruby, Go, Rust, PHP, Swift, Kotlin, TypeScript, Text, Markdown
                </div>
            </div>
        `;
        
        // Call clear endpoint
        await fetch('/clear', { method: 'POST' });
    }
}

// Focus input on load
window.addEventListener('load', () => {
    messageInput.focus();
});
