// Form validation and handling

class FormValidator {
    constructor() {
        this.init();
    }

    init() {
        this.setupFormValidation();
        this.setupRealTimeValidation();
    }

    setupFormValidation() {
        document.addEventListener('DOMContentLoaded', () => {
            const forms = document.querySelectorAll('form[data-validate]');
            
            forms.forEach(form => {
                form.addEventListener('submit', (e) => {
                    if (!this.validateForm(form)) {
                        e.preventDefault();
                        this.showFormErrors(form);
                    }
                });
            });
        });
    }

    setupRealTimeValidation() {
        document.addEventListener('input', (e) => {
            if (e.target.matches('input[data-validate], select[data-validate], textarea[data-validate]')) {
                this.validateField(e.target);
            }
        });

        document.addEventListener('blur', (e) => {
            if (e.target.matches('input[data-validate], select[data-validate], textarea[data-validate]')) {
                this.validateField(e.target);
            }
        });
    }

    validateForm(form) {
        let isValid = true;
        const fields = form.querySelectorAll('input[data-validate], select[data-validate], textarea[data-validate]');
        
        fields.forEach(field => {
            if (!this.validateField(field)) {
                isValid = false;
            }
        });

        return isValid;
    }

    validateField(field) {
        const value = field.value.trim();
        const validationRules = field.dataset.validate ? field.dataset.validate.split(' ') : [];
        let isValid = true;
        let errorMessage = '';

        // Clear previous error
        this.clearFieldError(field);

        // Required validation
        if (validationRules.includes('required') && !value) {
            isValid = false;
            errorMessage = this.getTranslation('field_required');
        }

        // Email validation
        if (isValid && validationRules.includes('email') && value) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(value)) {
                isValid = false;
                errorMessage = this.getTranslation('invalid_email');
            }
        }

        // Phone validation
        if (isValid && validationRules.includes('phone') && value) {
            const phoneRegex = /^[+][(]{0,1}[0-9]{1,4}[)]{0,1}[-\s\./0-9]$/;
            if (!phoneRegex.test(value)) {
                isValid = false;
                errorMessage = this.getTranslation('invalid_phone');
            }
        }

        // Minimum length validation
        if (isValid && validationRules.includes('minlength') && value) {
            const minLength = parseInt(field.dataset.minlength) || 6;
            if (value.length < minLength) {
                isValid = false;
                errorMessage = this.getTranslation('min_length').replace('{n}', minLength);
            }
        }

        // Maximum length validation
        if (isValid && validationRules.includes('maxlength') && value) {
            const maxLength = parseInt(field.dataset.maxlength) || 255;
            if (value.length > maxLength) {
                isValid = false;
                errorMessage = this.getTranslation('max_length').replace('{n}', maxLength);
            }
        }

        // Pattern validation
        if (isValid && validationRules.includes('pattern') && value) {
            const pattern = new RegExp(field.dataset.pattern);
            if (!pattern.test(value)) {
                isValid = false;
                errorMessage = field.dataset.patternMessage || this.getTranslation('invalid_pattern');
            }
        }

        // Numeric validation
        if (isValid && validationRules.includes('numeric') && value) {
            if (isNaN(value)) {
                isValid = false;
                errorMessage = this.getTranslation('numeric_only');
            }
        }

        if (!isValid) {
            this.showFieldError(field, errorMessage);
        } else {
            this.showFieldSuccess(field);
        }

        return isValid;
    }

    showFieldError(field, message) {
        field.classList.add('is-invalid');
        field.classList.remove('is-valid');

        let errorElement = field.nextElementSibling;
        if (!errorElement || !errorElement.classList.contains('invalid-feedback')) {
            errorElement = document.createElement('div');
            errorElement.className = 'invalid-feedback';
            field.parentNode.appendChild(errorElement);
        }

        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }

    showFieldSuccess(field) {
        field.classList.remove('is-invalid');
        field.classList.add('is-valid');

        const errorElement = field.nextElementSibling;
        if (errorElement && errorElement.classList.contains('invalid-feedback')) {
            errorElement.style.display = 'none';
        }
    }

    clearFieldError(field) {
        field.classList.remove('is-invalid');
        field.classList.remove('is-valid');

        const errorElement = field.nextElementSibling;
        if (errorElement && errorElement.classList.contains('invalid-feedback')) {
            errorElement.style.display = 'none';
        }
    }

    showFormErrors(form) {
        const firstInvalidField = form.querySelector('.is-invalid');
        if (firstInvalidField) {
            firstInvalidField.focus();
        }

        // Show general form error message
        this.showToast(this.getTranslation('form_errors'), 'error');
    }

    showToast(message, type = 'info') {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;

        const toastContainer = document.querySelector('.toast-container') || this.createToastContainer();
        toastContainer.appendChild(toast);

        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }

    createToastContainer() {
        const container = document.createElement('div');
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '1100';
        document.body.appendChild(container);
        return container;
    }

    getTranslation(key) {
        // Simple translation lookup - would integrate with translation manager
        const translations = {
            'field_required': 'This field is required',
            'invalid_email': 'Please enter a valid email address',
            'invalid_phone': 'Please enter a valid phone number',
            'min_length': 'Must be at least {n} characters',
            'max_length': 'Cannot exceed {n} characters',
            'invalid_pattern': 'Invalid format',
            'numeric_only': 'Must be a number',
            'form_errors': 'Please correct the errors in the form'
        };

        return translations[key] || 'Validation error';
    }
}

// Initialize form validator
document.addEventListener('DOMContentLoaded', () => {
    window.formValidator = new FormValidator();
});