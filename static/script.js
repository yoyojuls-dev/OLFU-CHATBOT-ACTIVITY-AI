// public/script.js
document.addEventListener('DOMContentLoaded', () => {
    // Chat elements
    const messagesDiv = document.getElementById('messages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');

    // Modal elements
    const nameModalOverlay = document.getElementById('nameModalOverlay');
    const nameInputModal = document.getElementById('nameInputModal');
    const submitNameButton = document.getElementById('submitNameButton');

    let currentSessionId = localStorage.getItem('chatSessionId');
    let userName = 'Guest'; // Default user name

    // ---- Helper Functions ----

    function addMessage(text, sender, isError = false) {
        // (Keep the existing addMessage function as it is)
        errorDiv.style.display = 'none';
        const messageElem = document.createElement('div');
        messageElem.classList.add('message', sender);
        if (isError) {
            messageElem.classList.add('error');
            messageElem.textContent = `⚠️ ${text}`;
        } else {
             messageElem.innerHTML = text
                .replace(/</g, "<")
                .replace(/>/g, ">")
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\n/g, '<br>');
        }
        messagesDiv.appendChild(messageElem);
        messagesDiv.scrollTo({ top: messagesDiv.scrollHeight, behavior: 'smooth' });
    }

    function displayError(message) {
         // (Keep the existing displayError function as it is)
        errorDiv.textContent = `⚠️ Error: ${message}`;
        errorDiv.style.display = 'block';
        messagesDiv.scrollTo({ top: messagesDiv.scrollHeight, behavior: 'smooth' });
    }

    function adjustTextareaHeight() {
         // (Keep the existing adjustTextareaHeight function as it is)
        userInput.style.height = 'auto';
        let scrollHeight = userInput.scrollHeight;
        const maxHeight = 120;
        userInput.style.height = Math.min(scrollHeight, maxHeight) + 'px';
    }

    // ---- Name Modal Logic ----

    function handleNameSubmit() {
        const nameInput = nameInputModal.value.trim();
        if (nameInput !== "") {
            userName = nameInput;
        } else {
            userName = "Guest"; // Fallback if empty
        }

        // Hide the modal
        nameModalOverlay.style.display = 'none';

        // Enable chat input
        userInput.disabled = false;
        sendButton.disabled = false;

        // Display personalized welcome message
        const welcomeText = `Hello, <strong>${userName}</strong>! I'm the chatbot for OLFU's Senior High School program at the Quezon City Campus. How can I assist you today? Feel free to ask about strands, admission requirements, facilities, contact details, or anything else related to SHS at OLFU-QC.`;
        addMessage(welcomeText, 'model'); // Or use a 'system' class if preferred

        // Focus the main chat input
        userInput.focus();
        adjustTextareaHeight(); // Adjust height in case browser pre-filled it
    }

    // Show the modal when the page is ready
    function showNameModal() {
        if (nameModalOverlay) { // Check if modal exists
             nameModalOverlay.style.display = 'flex';
             // Try to focus, might need a slight delay sometimes
             setTimeout(() => nameInputModal.focus(), 50);
        } else {
             console.error("Name modal overlay not found in HTML!");
             // Fallback if modal is missing: just enable chat
             userInput.disabled = false;
             sendButton.disabled = false;
             addMessage("Hello! How can I help you today?", 'model'); // Generic welcome
        }
    }

    // ---- Send Message Logic ----

    async function sendMessage() {
        // (Keep the existing sendMessage async function as it is)
        const messageText = userInput.value.trim();
        if (!messageText) return;

        addMessage(messageText, 'user');
        userInput.value = '';
        adjustTextareaHeight();
        loadingDiv.style.display = 'block';
        sendButton.disabled = true;
        userInput.disabled = true;
        errorDiv.style.display = 'none';

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    message: messageText,
                    sessionId: currentSessionId
                }),
            });

            if (!response.ok) {
                 let errorMsg = `Server responded with status: ${response.status}`;
                 try { const errorData = await response.json(); errorMsg = errorData.error || errorMsg; } catch(e) { /* Ignore */ }
                 throw new Error(errorMsg);
            }
            const contentType = response.headers.get("content-type");
            if (!contentType || !contentType.includes("application/json")) {
                 throw new Error(`Received non-JSON response.`);
            }
            const data = await response.json();
            if (data.error) { throw new Error(data.error); }

            addMessage(data.response, 'model');
            if (data.sessionId && data.sessionId !== currentSessionId) {
                currentSessionId = data.sessionId;
                localStorage.setItem('chatSessionId', currentSessionId);
                console.log("Session ID updated:", currentSessionId);
            }

        } catch (error) {
            console.error('Error sending message:', error);
            displayError(error.message || 'Could not connect.');
        } finally {
            loadingDiv.style.display = 'none';
            sendButton.disabled = false;
            userInput.disabled = false;
            userInput.focus();
        }
    }

    // --- Initial Setup & Event Listeners ---

    showNameModal(); // Show the modal on load

    // Modal listeners
    submitNameButton.addEventListener('click', handleNameSubmit);
    nameInputModal.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault(); // Prevent potential form submission if wrapped in form
            handleNameSubmit();
        }
    });

    // Chat listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
    userInput.addEventListener('input', adjustTextareaHeight);

}); // End DOMContentLoaded