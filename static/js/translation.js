// Translation functionality for multilingual support

class TranslationManager {
    constructor() {
        this.currentLanguage = 'en';
        this.translations = {
            'en': {
                'welcome': 'Welcome to FarmIQ',
                'dashboard': 'Dashboard',
                'weather': 'Weather',
                'soil_health': 'Soil Health',
                'pest_risk': 'Pest Risk',
                'recommendations': 'Recommendations',
                'irrigation': 'Irrigation',
                'fertilizer': 'Fertilizer',
                'sowing': 'Sowing',
                'sustainability': 'Sustainability',
                'profile': 'Profile',
                'logout': 'Logout',
                'temperature': 'Temperature',
                'humidity': 'Humidity',
                'rainfall': 'Rainfall',
                'nitrogen': 'Nitrogen',
                'phosphorus': 'Phosphorus',
                'potassium': 'Potassium',
                'ph_level': 'pH Level',
                'organic_matter': 'Organic Matter',
                'current': 'Current',
                'optimal': 'Optimal',
                'low_risk': 'Low Risk',
                'medium_risk': 'Medium Risk',
                'high_risk': 'High Risk',
                'voice_assistant': 'Voice Assistant',
                'listen': 'Listen',
                'stop': 'Stop',
                'english': 'English',
                'hindi': 'Hindi',
                'notifications': 'Notifications',
                'settings': 'Settings',
                'help': 'Help',
                'contact': 'Contact',
                'about': 'About'
            },
            'hi': {
                'welcome': 'फार्मIQ में आपका स्वागत है',
                'dashboard': 'डैशबोर्ड',
                'weather': 'मौसम',
                'soil_health': 'मिट्टी का स्वास्थ्य',
                'pest_risk': 'कीट जोखिम',
                'recommendations': 'सिफारिशें',
                'irrigation': 'सिंचाई',
                'fertilizer': 'उर्वरक',
                'sowing': 'बुवाई',
                'sustainability': 'स्थिरता',
                'profile': 'प्रोफाइल',
                'logout': 'लॉगआउट',
                'temperature': 'तापमान',
                'humidity': 'नमी',
                'rainfall': 'वर्षा',
                'nitrogen': 'नाइट्रोजन',
                'phosphorus': 'फास्फोरस',
                'potassium': 'पोटैशियम',
                'ph_level': 'pH स्तर',
                'organic_matter': 'जैविक पदार्थ',
                'current': 'वर्तमान',
                'optimal': 'इष्टतम',
                'low_risk': 'कम जोखिम',
                'medium_risk': 'मध्यम जोखिम',
                'high_risk': 'उच्च जोखिम',
                'voice_assistant': 'वॉयस असिस्टेंट',
                'listen': 'सुनें',
                'stop': 'रोकें',
                'english': 'अंग्रेजी',
                'hindi': 'हिंदी',
                'notifications': 'सूचनाएं',
                'settings': 'सेटिंग्स',
                'help': 'मदद',
                'contact': 'संपर्क',
                'about': 'के बारे में'
            }
        };
        this.init();
    }

    init() {
        this.loadLanguagePreference();
        this.applyTranslations();
        this.setupLanguageToggle();
    }

    loadLanguagePreference() {
        const savedLanguage = localStorage.getItem('farmiq_language');
        if (savedLanguage && this.translations[savedLanguage]) {
            this.currentLanguage = savedLanguage;
        }
    }

    applyTranslations() {
        // Translate all elements with data-translate attribute
        const elements = document.querySelectorAll('[data-translate]');
        elements.forEach(element => {
            const key = element.getAttribute('data-translate');
            const translation = this.getTranslation(key);
            if (translation) {
                if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                    element.placeholder = translation;
                } else {
                    element.textContent = translation;
                }
            }
        });

        // Update language toggle UI
        this.updateLanguageToggleUI();
    }

    getTranslation(key) {
        return this.translations[this.currentLanguage]?.[key] || this.translations['en'][key];
    }

    setLanguage(language) {
        if (this.translations[language]) {
            this.currentLanguage = language;
            localStorage.setItem('farmiq_language', language);
            this.applyTranslations();
            
            // Dispatch event for other components
            document.dispatchEvent(new CustomEvent('languageChanged', {
                detail: { language: language }
            }));
        }
    }

    setupLanguageToggle() {
        const toggleButtons = document.querySelectorAll('[data-lang-toggle]');
        toggleButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const language = e.target.getAttribute('data-lang-toggle');
                this.setLanguage(language);
            });
        });
    }

    updateLanguageToggleUI() {
        const buttons = document.querySelectorAll('[data-lang-toggle]');
        buttons.forEach(button => {
            const language = button.getAttribute('data-lang-toggle');
            button.classList.toggle('active', language === this.currentLanguage);
        });
    }

    // Method to add dynamic translations
    addTranslation(language, key, translation) {
        if (!this.translations[language]) {
            this.translations[language] = {};
        }
        this.translations[language][key] = translation;
    }

    // Method to translate dynamic content
    translateDynamicContent(text) {
        // Simple implementation - in real app, you'd have a more sophisticated approach
        for (const [key, translation] of Object.entries(this.translations[this.currentLanguage])) {
            if (text.includes(key)) {
                text = text.replace(key, translation);
            }
        }
        return text;
    }
}

// Initialize translation manager
document.addEventListener('DOMContentLoaded', () => {
    window.translationManager = new TranslationManager();
});