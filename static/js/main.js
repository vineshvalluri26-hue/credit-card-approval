/* ==========================================================================
   VaultGate AI - Frontend JavaScript Logic
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Mobile Drawer Navigation
    initMobileMenu();

    // Form Event Listeners (apply page)
    initFormInteractions();

    // Result gauge loading animation (result page)
    initResultGauge();

    // Fix number inputs - prevent scroll wheel and arrow key changes
    fixNumberInputs();
});

/**
 * Handles the mobile responsive header menu toggle drawer
 */
function initMobileMenu() {
    const menuToggle = document.querySelector('.mobile-nav-toggle');
    const menuClose = document.querySelector('.mobile-nav-close');
    const sidebar = document.querySelector('.mobile-sidebar');
    
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            sidebar.classList.add('open');
        });
    }

    if (menuClose && sidebar) {
        menuClose.addEventListener('click', () => {
            sidebar.classList.remove('open');
        });
    }

    // Close mobile menu if clicked outside
    document.addEventListener('click', (e) => {
        if (sidebar && sidebar.classList.contains('open') && !sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
            sidebar.classList.remove('open');
        }
    });
}

/**
 * Fix all number-type inputs: prevent scroll wheel from changing values
 * and prevent unintentional value changes
 */
function fixNumberInputs() {
    // Prevent scroll wheel from changing number input values
    document.querySelectorAll('input[type="number"]').forEach(input => {
        input.addEventListener('wheel', (e) => {
            e.preventDefault();
            input.blur();
        }, { passive: false });
    });

    // For text inputs with inputmode="numeric" - validate on form submit
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', (e) => {
            // Validate numeric text inputs
            const numericInputs = form.querySelectorAll('input[inputmode="numeric"], input[inputmode="decimal"]');
            for (const input of numericInputs) {
                const val = input.value.trim();
                if (val === '' || isNaN(parseFloat(val))) {
                    e.preventDefault();
                    input.focus();
                    input.style.borderColor = '#f43f5e';
                    input.style.boxShadow = '0 0 12px rgba(244, 63, 94, 0.4)';
                    setTimeout(() => {
                        input.style.borderColor = '';
                        input.style.boxShadow = '';
                    }, 2000);
                    return false;
                }
            }
        });
    }
}

/**
 * Handles dynamic interactions within the application forms (apply page)
 */
function initFormInteractions() {
    // Co-dependency between Retirement status and Employment years
    const retiredCheckbox = document.getElementById('is_retired');
    const employmentInput = document.getElementById('employment_years');
    const incomeTypeSelect = document.getElementById('income_type');

    if (retiredCheckbox && employmentInput) {
        // Toggle input disable based on Retired status
        retiredCheckbox.addEventListener('change', () => {
            if (retiredCheckbox.checked) {
                employmentInput.value = 0;
                employmentInput.disabled = true;
                employmentInput.style.opacity = '0.5';
                
                // Auto switch income type to Pensioner if not already
                if (incomeTypeSelect && incomeTypeSelect.value !== 'Pensioner') {
                    incomeTypeSelect.value = 'Pensioner';
                }
            } else {
                employmentInput.disabled = false;
                employmentInput.style.opacity = '1';
                
                // Switch income type off Pensioner if it was set
                if (incomeTypeSelect && incomeTypeSelect.value === 'Pensioner') {
                    incomeTypeSelect.value = 'Working';
                }
            }
        });

        // Sync drop-down updates to retired status
        if (incomeTypeSelect) {
            incomeTypeSelect.addEventListener('change', () => {
                if (incomeTypeSelect.value === 'Pensioner') {
                    retiredCheckbox.checked = true;
                    retiredCheckbox.dispatchEvent(new Event('change'));
                } else if (retiredCheckbox.checked) {
                    retiredCheckbox.checked = false;
                    retiredCheckbox.dispatchEvent(new Event('change'));
                }
            });
        }
        
        // Trigger check on load
        if (retiredCheckbox.checked || (incomeTypeSelect && incomeTypeSelect.value === 'Pensioner')) {
            retiredCheckbox.checked = true;
            employmentInput.value = 0;
            employmentInput.disabled = true;
            employmentInput.style.opacity = '0.5';
        }
    }
}

/**
 * Reads confidence score attributes on prediction result load to run animation gauge
 */
function initResultGauge() {
    const gaugeContainer = document.querySelector('.circular-progress.animate-on-load');
    const scoreVal = document.getElementById('score-percentage');

    if (gaugeContainer && scoreVal) {
        const targetScore = parseInt(gaugeContainer.getAttribute('data-score') || '0', 10);
        const prediction = parseInt(gaugeContainer.getAttribute('data-prediction') || '0', 10);
        
        // Trigger conic gradient progress animation
        animateProgress(targetScore, prediction, gaugeContainer, scoreVal);
    }
}

/**
 * Animates the circular confidence ring representation
 */
function animateProgress(targetScore, prediction, containerEl, textEl) {
    let currentScore = 0;
    const speed = 10; // lower is faster
    
    // Choose ring color based on status (0=Approved: Royal Purple/Violet, 1=Rejected: Crimson Pink)
    const progressColor = prediction === 0 ? '#8b5cf6' : '#f43f5e';
    const trackColor = 'rgba(255, 255, 255, 0.08)';

    const timer = setInterval(() => {
        if (targetScore <= 0) {
            containerEl.style.background = `conic-gradient(${progressColor} 0deg, ${trackColor} 0deg)`;
            textEl.textContent = `0%`;
            clearInterval(timer);
            return;
        }

        currentScore++;
        
        // Update conic gradient degrees
        containerEl.style.background = `conic-gradient(${progressColor} ${currentScore * 3.6}deg, ${trackColor} ${currentScore * 3.6}deg)`;
        textEl.textContent = `${currentScore}%`;
        
        if (currentScore >= targetScore) {
            clearInterval(timer);
        }
    }, speed);
}
