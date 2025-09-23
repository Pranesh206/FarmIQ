// [DEV] Enhanced dashboard JS with console logging
console.log("[DEV] Dev dashboard loaded.");

// Voice TTS (same as dashboard.js, but with log)
document.getElementById('voice-advice').addEventListener('click', function() {
    const advice = document.querySelector('ul').innerText;
    const utterance = new SpeechSynthesisUtterance(advice);
    speechSynthesis.speak(utterance);
    console.log("[DEV] Voice prompt triggered:", advice);
});

// Language Toggle (Mock)
document.getElementById('toggle-lang').addEventListener('click', function() {
    const isHindi = this.textContent.includes('Hindi');
    if (isHindi) {
        // Mock translate to English
        document.querySelector('h3').innerText = 'Current Weather: 32°C, Rainfall: 150mm';  // Reset
        this.textContent = 'Switch to Hindi';
    } else {
        // Mock Hindi (placeholder)
        document.querySelector('h3').innerText = 'वर्तमान मौसम: 32°C, वर्षा: 150mm';
        this.textContent = 'Switch to English';
    }
    console.log("[DEV] Language toggled.");
});

// Simple Chart (using Chart.js CDN in base.html