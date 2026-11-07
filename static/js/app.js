document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    const sections = {
        hero: document.getElementById('hero'),
        input: document.getElementById('input-section'),
        loading: document.getElementById('loading-section'),
        results: document.getElementById('results-section')
    };

    function showSection(sectionId) {
        Object.values(sections).forEach(s => s.classList.remove('active'));
        sections[sectionId].classList.add('active');
    }

    // Buttons
    document.getElementById('upload-btn').addEventListener('click', () => {
        showSection('input');
        document.getElementById('upload-zone').classList.remove('hidden');
        document.getElementById('camera-zone').classList.add('hidden');
    });

    document.getElementById('camera-btn').addEventListener('click', () => {
        showSection('input');
        document.getElementById('upload-zone').classList.add('hidden');
        document.getElementById('camera-zone').classList.remove('hidden');
        startCamera();
    });

    document.getElementById('back-to-hero').addEventListener('click', () => showSection('hero'));

    // File Upload
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');

    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--primary)';
    });
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = 'var(--glass-border)';
    });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    fileInput.addEventListener('change', (e) => {
        handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file) return;
        const formData = new FormData();
        formData.append('image', file);
        performAnalysis(formData);
    }

    // Camera Logic
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    const postCapture = document.getElementById('post-capture');
    const retakeBtn = document.getElementById('retake-btn');
    const analyzeBtn = document.getElementById('analyze-btn');

    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
            video.srcObject = stream;
        } catch (err) {
            alert('Camera access denied or unavailable.');
            showSection('hero');
        }
    }

    captureBtn.addEventListener('click', () => {
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
            window.capturedFile = file;

            const stream = video.srcObject;
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }

            video.classList.add('hidden');
            captureBtn.classList.add('hidden');
            postCapture.classList.remove('hidden');
        }, 'image/jpeg');
    });

    retakeBtn.addEventListener('click', () => {
        video.classList.remove('hidden');
        captureBtn.classList.remove('hidden');
        postCapture.classList.add('hidden');
        startCamera();
    });

    analyzeBtn.addEventListener('click', () => {
        const formData = new FormData();
        formData.append('image', window.capturedFile);
        performAnalysis(formData);
    });

    // Analysis Pipeline
    let latestMetrics = null;
    const chatHistory = [];
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatSend = document.getElementById('chat-send');
    const chatPanel = document.getElementById('chat-panel');
    const chatToggle = document.getElementById('chat-toggle');

    chatToggle.addEventListener('click', () => {
        chatPanel.classList.remove('hidden');
        chatToggle.setAttribute('aria-expanded', 'true');
        chatInput.focus();
    });

    document.getElementById('chat-close').addEventListener('click', () => {
        chatPanel.classList.add('hidden');
        chatToggle.setAttribute('aria-expanded', 'false');
        chatToggle.focus();
    });

    function addChatMessage(role, content) {
        const message = document.createElement('div');
        message.className = `chat-message ${role}-message`;
        message.textContent = content;
        chatMessages.appendChild(message);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return message;
    }

    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const message = chatInput.value.trim();
        if (!message || chatSend.disabled) return;

        addChatMessage('user', message);
        chatInput.value = '';
        chatInput.disabled = true;
        chatSend.disabled = true;
        const pendingMessage = addChatMessage('assistant', 'Thinking...');

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    history: chatHistory.slice(-10),
                    skin_context: latestMetrics
                })
            });
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || 'The assistant could not answer right now.');
            }

            pendingMessage.textContent = data.reply;
            chatHistory.push(
                { role: 'user', content: message },
                { role: 'assistant', content: data.reply }
            );
        } catch (error) {
            pendingMessage.textContent = error.message;
            pendingMessage.classList.add('chat-error');
        } finally {
            chatInput.disabled = false;
            chatSend.disabled = false;
            chatInput.focus();
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    });

    async function performAnalysis(formData) {
        showSection('loading');
        const steps = [
            document.getElementById('step-0'),
            document.getElementById('step-1'),
            document.getElementById('step-2'),
            document.getElementById('step-3'),
            document.getElementById('step-4')
        ];

        async function updateStep(index, status) {
            if (status === 'active') {
                steps[index].classList.add('active');
                steps[index].classList.remove('completed');
            } else if (status === 'completed') {
                steps[index].classList.remove('active');
                steps[index].classList.add('completed');
            }
        }

        try {
            // Simulate pipeline steps for UX
            await updateStep(0, 'active');
            await new Promise(r => setTimeout(r, 600));
            await updateStep(0, 'completed');

            await updateStep(1, 'active');
            await new Promise(r => setTimeout(r, 600));
            await updateStep(1, 'completed');

            await updateStep(2, 'active');
            await new Promise(r => setTimeout(r, 600));
            await updateStep(2, 'completed');

            await updateStep(3, 'active');
            await new Promise(r => setTimeout(r, 600));
            await updateStep(3, 'completed');

            await updateStep(4, 'active');

            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (!data.success) throw new Error(data.error);

            await updateStep(4, 'completed');
            displayResults(data);
            showSection('results');

        } catch (err) {
            alert('Analysis failed: ' + err.message);
            showSection('hero');
        }
    }

    function displayResults(data) {
        document.getElementById('res-image').src = data.annotated_image;
        chatToggle.classList.remove('hidden');

        const m = data.metrics;
        latestMetrics = {
            acne_like_regions: m.acne_like_regions,
            redness_score: m.redness_score,
            dark_spots: m.dark_spots,
            pore_like_regions: m.pore_like_regions,
            skin_smoothness: m.skin_smoothness
        };
        document.getElementById('metric-redness').style.width = (m.redness_score * 100) + '%';
        document.getElementById('metric-smoothness').style.width = m.skin_smoothness + '%';
        document.getElementById('metric-uniformity').style.width = m.skin_uniformity + '%';
        document.getElementById('metric-oiliness').style.width = (m.oiliness_score * 100) + '%';

        document.getElementById('metric-spots').textContent = m.dark_spots;
        document.getElementById('metric-acne').textContent = m.acne_like_regions;
        document.getElementById('metric-pores').textContent = m.pore_like_regions;

        const badge = document.getElementById('attention-badge');
        badge.textContent = data.attention_level;
        badge.className = 'attention-badge ' + data.attention_level.split(' ')[0].toLowerCase();

        document.getElementById('ai-text').textContent = data.ai_interpretation;

        // Routine
        const amList = document.getElementById('routine-am');
        const pmList = document.getElementById('routine-pm');
        amList.innerHTML = '';
        pmList.innerHTML = '';

        data.routine.am.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            amList.appendChild(li);
        });
        data.routine.pm.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            pmList.appendChild(li);
        });
    }
});
