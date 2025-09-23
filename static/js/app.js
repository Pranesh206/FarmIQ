// FarmIQ Main Application JavaScript

class FarmIQApp {
    constructor() {
        this.currentLanguage = 'en';
        this.isVoiceActive = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadUserPreferences();
        this.initializeCharts();
        this.checkBrowserCompatibility();
    }

    setupEventListeners() {
        // Language toggle
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.toggleLanguage(e.target.dataset.lang);
            });
        });

        // Voice controls
        document.querySelectorAll('.voice-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.toggleVoiceAssistant();
            });
        });

        // Form submissions
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', (e) => {
                this.handleFormSubmit(e);
            });
        });

        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                this.handleNavigation(e);
            });
        });

        // Responsive menu
        const menuToggle = document.querySelector('.navbar-toggler');
        if (menuToggle) {
            menuToggle.addEventListener('click', () => {
                this.toggleMobileMenu();
            });
        }
    }

    toggleLanguage(lang) {
        if (this.currentLanguage !== lang) {
            this.currentLanguage = lang;
            this.updateLanguageUI();
            this.savePreferences();
            // Dispatch custom event for other components
            document.dispatchEvent(new CustomEvent('languageChanged', {
                detail: { language: lang }
            }));
        }
    }

    updateLanguageUI() {
        // Update active language button
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.lang === this.currentLanguage);
        });

        // Update content based on language (simplified implementation)
        this.translateContent();
    }

    translateContent() {
        // This would typically integrate with a translation service
        const elements = document.querySelectorAll('[data-translate]');
        elements.forEach(el => {
            const key = el.dataset.translate;
            // In a real implementation, this would fetch from translation files
            const translations = {
                'welcome': {
                    'en': 'Welcome to FarmIQ',
                    'hi': 'फार्मIQ में आपका स्वागत है'
                },
                'dashboard': {
                    'en': 'Dashboard',
                    'hi': 'डैशबोर्ड'
                }
                // Add more translations as needed
            };
            
            if (translations[key] && translations[key][this.currentLanguage]) {
                el.textContent = translations[key][this.currentLanguage];
            }
        });
    }

    toggleVoiceAssistant() {
        this.isVoiceActive = !this.isVoiceActive;
        const voiceBtn = document.querySelector('.voice-btn');
        
        if (this.isVoiceActive) {
            voiceBtn.classList.add('active');
            this.startVoiceAssistant();
        } else {
            voiceBtn.classList.remove('active');
            this.stopVoiceAssistant();
        }
    }

    startVoiceAssistant() {
        // Initialize voice recognition
        if ('speechSynthesis' in window && 'SpeechRecognition' in window) {
            this.speak('Voice assistant activated. How can I help you?');
        } else {
            this.showNotification('Voice features not supported in your browser', 'warning');
            this.isVoiceActive = false;
            document.querySelector('.voice-btn').classList.remove('active');
        }
    }

    stopVoiceAssistant() {
        if (window.speechSynthesis) {
            window.speechSynthesis.cancel();
        }
    }

    speak(text) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = this.currentLanguage === 'hi' ? 'hi-IN' : 'en-US';
            window.speechSynthesis.speak(utterance);
        }
    }

    handleFormSubmit(e) {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData(form);
        
        // Basic validation
        if (!this.validateForm(form)) {
            this.showNotification('Please fill all required fields correctly', 'warning');
            return;
        }

        // Show loading state
        this.setLoadingState(form, true);

        // Submit form (would typically be an AJAX call)
        setTimeout(() => {
            this.setLoadingState(form, false);
            this.showNotification('Form submitted successfully', 'success');
            form.reset();
        }, 1000);
    }

    validateForm(form) {
        let isValid = true;
        const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
        
        inputs.forEach(input => {
            if (!input.value.trim()) {
                isValid = false;
                input.classList.add('is-invalid');
            } else {
                input.classList.remove('is-invalid');
            }
        });

        return isValid;
    }

    setLoadingState(form, isLoading) {
        const button = form.querySelector('button[type="submit"]');
        if (isLoading) {
            button.disabled = true;
            button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
        } else {
            button.disabled = false;
            button.innerHTML = button.dataset.originalText || 'Submit';
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show`;
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Add to notification container or create one
        let container = document.querySelector('.notification-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'notification-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '1050';
            document.body.appendChild(container);
        }
        
        container.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 150);
        }, 5000);
    }

    handleNavigation(e) {
        // Smooth scrolling for anchor links
        if (e.target.hash) {
            e.preventDefault();
            const target = document.querySelector(e.target.hash);
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        }
    }

    toggleMobileMenu() {
        const navbarCollapse = document.querySelector('.navbar-collapse');
        navbarCollapse.classList.toggle('show');
    }

    loadUserPreferences() {
        // Load saved preferences from localStorage
        const savedLang = localStorage.getItem('farmiq_language');
        const savedVoice = localStorage.getItem('farmiq_voice_enabled');
        
        if (savedLang) this.toggleLanguage(savedLang);
        if (savedVoice === 'true') this.toggleVoiceAssistant();
    }

    savePreferences() {
        localStorage.setItem('farmiq_language', this.currentLanguage);
        localStorage.setItem('farmiq_voice_enabled', this.isVoiceActive.toString());
    }

    initializeCharts() {
        // Initialize charts if Chart.js is available
        if (typeof Chart !== 'undefined') {
            setTimeout(() => {
                if (typeof initializeCharts === 'function') {
                    initializeCharts();
                }
            }, 100);
        }
    }

    checkBrowserCompatibility() {
        // Check for essential features
        const missingFeatures = [];
        
        if (!('localStorage' in window)) missingFeatures.push('Local Storage');
        if (!('speechSynthesis' in window)) missingFeatures.push('Text-to-Speech');
        if (!('SpeechRecognition' in window)) missingFeatures.push('Speech Recognition');
        
        if (missingFeatures.length > 0) {
            this.showNotification(
                `Some features may not work: ${missingFeatures.join(', ')}`,
                'warning'
            );
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.farmIQ = new FarmIQApp();
});