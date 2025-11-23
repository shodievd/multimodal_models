let currentImagePath = null;

// VQA Logic
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            document.getElementById('image-preview').src = e.target.result;
            document.getElementById('image-preview-container').style.display = 'inline-block';
            document.getElementById('send-btn').disabled = false;
        };
        reader.readAsDataURL(file);
        currentImagePath = null; // Reset server path since we have a new local file
    }
}

function clearImage() {
    document.getElementById('image-upload').value = '';
    document.getElementById('image-preview-container').style.display = 'none';
    currentImagePath = null;
    checkInput();
}

function checkInput() {
    const input = document.getElementById('chat-input').value;
    const hasImage = document.getElementById('image-upload').files.length > 0 || currentImagePath;
    document.getElementById('send-btn').disabled = !(input.trim() || hasImage);
}

document.getElementById('chat-input')?.addEventListener('input', checkInput);

async function handleChatSubmit(event) {
    event.preventDefault();

    const input = document.getElementById('chat-input');
    const fileInput = document.getElementById('image-upload');
    const prompt = input.value.trim();

    if (!prompt && !fileInput.files.length && !currentImagePath) return;

    // Add user message to chat
    addMessage('user', prompt, fileInput.files[0] ? URL.createObjectURL(fileInput.files[0]) : (currentImagePath ? document.getElementById('image-preview').src : null));

    const formData = new FormData();
    formData.append('prompt', prompt);

    // Add model selection
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        formData.append('model_size', modelSelect.value);
    }

    if (fileInput.files.length > 0) {
        formData.append('image', fileInput.files[0]);
    } else if (currentImagePath) {
        formData.append('image_path', currentImagePath);
    } else {
        addMessage('system', 'Please upload an image first.');
        return;
    }

    // Clear input but keep image for follow-up
    input.value = '';
    document.getElementById('send-btn').disabled = true;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            addMessage('assistant', data.response);
            currentImagePath = data.image_path; // Update server path
        } else {
            addMessage('system', 'Error: ' + data.error);
        }
    } catch (error) {
        addMessage('system', 'Error: ' + error.message);
    } finally {
        document.getElementById('send-btn').disabled = false;
    }
}

function addMessage(role, text, imageUrl = null) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    let contentHtml = '';
    if (imageUrl && role === 'user') {
        const fileInput = document.getElementById('image-upload');
        // Only show image if it was just uploaded (simple check)
        if (fileInput.files.length > 0 || !currentImagePath) {
            contentHtml += `<img src="${imageUrl}" class="message-image">`;
        }
        // We don't clear file input here anymore to allow follow up with same image easily
        // But we need to handle the display logic better. 
        // For this demo, let's just show it if passed.
    }

    contentHtml += `<div class="message-content">${text}</div>`;
    messageDiv.innerHTML = contentHtml;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Generate Logic
async function handleGenerateSubmit(event) {
    event.preventDefault();

    const promptInput = document.getElementById('prompt');
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    const btn = document.getElementById('generate-btn');
    const placeholder = document.getElementById('result-placeholder');
    const spinner = document.getElementById('loading-spinner');
    const resultContainer = document.getElementById('generated-image-container');
    const resultImage = document.getElementById('generated-image');
    const downloadBtn = document.getElementById('download-btn');

    btn.disabled = true;
    placeholder.style.display = 'none';
    resultContainer.style.display = 'none';
    spinner.style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('prompt', prompt);

        const response = await fetch('/api/generate', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            resultImage.src = data.image_url;
            downloadBtn.href = data.image_url;
            resultContainer.style.display = 'flex';
        } else {
            placeholder.style.display = 'flex';
            placeholder.innerHTML = `<p style="color: #ef4444">Error: ${data.error}</p>`;
        }
    } catch (error) {
        placeholder.style.display = 'flex';
        placeholder.innerHTML = `<p style="color: #ef4444">Error: ${error.message}</p>`;
    } finally {
        spinner.style.display = 'none';
        btn.disabled = false;
    }
}

// Settings Logic
function openSettings() {
    document.getElementById('settings-modal').style.display = 'block';
    fetchConfig();
}

function closeSettings() {
    document.getElementById('settings-modal').style.display = 'none';
}

async function fetchConfig() {
    try {
        const response = await fetch('/api/config');
        const data = await response.json();

        const settingsSelect = document.getElementById('device-select');
        const chatSelect = document.getElementById('device-select-chat');

        if (settingsSelect) settingsSelect.value = data.device;
        if (chatSelect) chatSelect.value = data.device;

    } catch (error) {
        console.error('Error fetching config:', error);
    }
}

async function updateDeviceFromChat() {
    const device = document.getElementById('device-select-chat').value;
    const settingsSelect = document.getElementById('device-select');
    if (settingsSelect) settingsSelect.value = device;

    await updateDeviceInternal(device);
}

async function updateDevice() {
    const device = document.getElementById('device-select').value;
    const chatSelect = document.getElementById('device-select-chat');
    if (chatSelect) chatSelect.value = device;

    await updateDeviceInternal(device);
}

async function updateDeviceInternal(device) {
    const status = document.getElementById('settings-status');
    if (status) {
        status.textContent = 'Updating...';
        status.style.color = '#fbbf24';
    }

    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ device: device })
        });

        const data = await response.json();

        if (response.ok) {
            if (status) {
                status.textContent = 'Updated successfully!';
                status.style.color = '#4ade80';
            }
        } else {
            if (status) {
                status.textContent = 'Error: ' + data.error;
                status.style.color = '#ef4444';
            }
            alert('Error switching device: ' + data.error);
            fetchConfig();
        }
    } catch (error) {
        if (status) {
            status.textContent = 'Error: ' + error.message;
            status.style.color = '#ef4444';
        }
        console.error('Error switching device:', error);
    }
}

// Close modal when clicking outside
window.onclick = function (event) {
    const modal = document.getElementById('settings-modal');
    if (event.target == modal) {
        modal.style.display = "none";
    }
}

// Init
document.addEventListener('DOMContentLoaded', fetchConfig);
