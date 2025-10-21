// Auto-resize textarea
const messageInput = document.getElementById('messageInput');
const chatContainer = document.getElementById('chatContainer');
const sendBtn = document.getElementById('sendBtn');

messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

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
            sourcesHTML += `<div class="source-item">
                ${index + 1}. ${source.source} (Page ${source.page}) - Relevance: ${(1 - source.score).toFixed(2)}
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
        // Clear chat container
        chatContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">💡</div>
                <h2>Welcome to 168!</h2>
                <p>I'm your AI coding assistant powered by RAG technology. I can help you with:</p>
                <div class="feature-grid">
                    <div class="feature-item">
                        <span class="feature-icon">🐍</span>
                        <span>Python Programming</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">☕</span>
                        <span>Java Development</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">📚</span>
                        <span>Code Examples</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">🔍</span>
                        <span>Best Practices</span>
                    </div>
                </div>
                <p class="welcome-subtext">Ask me anything about coding based on your learning materials!</p>
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
