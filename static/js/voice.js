// Voice assistant functionality

class VoiceAssistant {
    constructor() {
        this.recognition = null;
        this.isListening = false;
        this.commands = {
            'weather': () => this.handleWeatherCommand(),
            'soil': () => this.handleSoilCommand(),
            'pest': () => this.handlePestCommand(),
            'irrigation': () => this.handleIrrigationCommand(),
            'help': () => this.handleHelpCommand()
        };
        this.init();
    }

    init() {
        if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.lang = 'en-US';
            this.recognition.interimResults = false;
            this.recognition.maxAlternatives = 1;

            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript.toLowerCase();
                this.processCommand(transcript);
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.showVoiceFeedback('Sorry, I didn\'t catch that. Please try again.');
            };

            this.recognition.onend = () => {
                this.isListening = false;
                this.updateListeningUI();
            };
        }
    }

    startListening() {
        if (this.recognition) {
            this.recognition.start();
            this.isListening = true;
            this.updateListeningUI();
            this.showVoiceFeedback('Listening...');
        } else {
            this.showVoiceFeedback('Voice recognition not supported in your browser');
        }
    }

    stopListening() {
        if (this.recognition) {
            this.recognition.stop();
            this.isListening = false;
            this.updateListeningUI();
        }
    }

    processCommand(transcript) {
        let commandHandled = false;
        
        for (const [keyword, handler] of Object.entries(this.commands)) {
            if (transcript.includes(keyword)) {
                handler();
                commandHandled = true;
                break;
            }
        }

        if (!commandHandled) {
            this.showVoiceFeedback('I didn\'t understand that command. Say "help" for available commands.');
        }
    }

    handleWeatherCommand() {
        const weatherData = this.getWeatherData();
        this.speak(`Current weather: Temperature ${weatherData.temperature} degrees, ${weatherData.condition}. Humidity is ${weatherData.humidity} percent.`);
    }

    handleSoilCommand() {
        const soilData = this.getSoilData();
        this.speak(`Soil health: Nitrogen ${soilData.nitrogen} percent, Phosphorus ${soilData.phosphorus} percent, Potassium ${soilData.potassium} percent. pH level is ${soilData.pH}.`);
    }

    handlePestCommand() {
        const pestData = this.getPestData();
        this.speak(`Pest risk is currently ${pestData.riskLevel}. ${pestData.recommendation}`);
    }

    handleIrrigationCommand() {
        const irrigationData = this.getIrrigationData();
        this.speak(`Soil moisture is ${irrigationData.moisture} percent. ${irrigationData.recommendation}`);
    }

    handleHelpCommand() {
        const commandsList = Object.keys(this.commands).join(', ');
        this.speak(`Available commands: ${commandsList}. You can ask about weather, soil, pests, or irrigation.`);
    }

    getWeatherData() {
        // Mock data - would come from actual API in production
        return {
            temperature: 28,
            condition: 'partly cloudy',
            humidity: 65,
            rainfall: 0
        };
    }

    getSoilData() {
        // Mock data
        return {
            nitrogen: 75,
            phosphorus: 60,
            potassium: 85,
            pH: 6.8,
            organicMatter: 2.5
        };
    }

    getPestData() {
        // Mock data
        return {
            riskLevel: 'medium',
            recommendation: 'Monitor crops closely and consider preventive measures.'
        };
    }

    getIrrigationData() {
        // Mock data
        return {
            moisture: 45,
            recommendation: 'Irrigation is recommended within the next 24 hours.'
        };
    }

    speak(text) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
        }
    }

    showVoiceFeedback(message) {
        // Create or update voice feedback element
        let feedbackEl = document.getElementById('voice-feedback');
        if (!feedbackEl) {
            feedbackEl = document.createElement('div');
            feedbackEl.id = 'voice-feedback';
            feedbackEl.className = 'voice-feedback alert alert-info';
            feedbackEl.style.position = 'fixed';
            feedbackEl.style.bottom = '20px';
            feedbackEl.style.right = '20px';
            feedbackEl.style.zIndex = '1000';
            document.body.appendChild(feedbackEl);
        }
        
        feedbackEl.textContent = message;
        feedbackEl.classList.add('show');
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            feedbackEl.classList.remove('show');
        }, 3000);
    }

    updateListeningUI() {
        const voiceBtn = document.querySelector('.voice-btn');
        if (voiceBtn) {
            if (this.isListening) {
                voiceBtn.innerHTML = '<i class="fas fa-microphone-slash"></i>';
                voiceBtn.classList.add('listening');
            } else {
                voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
                voiceBtn.classList.remove('listening');
            }
        }
    }
}

// Initialize voice assistant
document.addEventListener('DOMContentLoaded', () => {
    window.voiceAssistant = new VoiceAssistant();
});