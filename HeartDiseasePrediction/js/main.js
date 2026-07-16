document.addEventListener('DOMContentLoaded', () => {
    // 1. Loader Logic
    const loader = document.getElementById('page-loader');
    if (loader) {
        // Fade out loader on DOM load complete
        setTimeout(() => {
            loader.style.opacity = '0';
            loader.style.visibility = 'hidden';
        }, 300);
    }

    // 2. Theme Switching Logic
    const themeToggleBtn = document.getElementById('theme-toggle');
    const currentTheme = localStorage.getItem('theme') || 'light';
    
    // Apply initial theme
    document.documentElement.setAttribute('data-theme', currentTheme);
    updateThemeIcon(currentTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const activeTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = activeTheme === 'dark' ? 'light' : 'dark';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeIcon(newTheme);
            
            showToast(`Theme switched to ${newTheme} mode.`, 'info');
        });
    }

    function updateThemeIcon(theme) {
        if (!themeToggleBtn) return;
        const icon = themeToggleBtn.querySelector('i');
        if (icon) {
            if (theme === 'dark') {
                icon.className = 'bi bi-sun-fill';
            } else {
                icon.className = 'bi bi-moon-stars-fill';
            }
        }
    }

    // 3. Scroll to Top Logic
    const scrollTopBtn = document.getElementById('scroll-top-btn');
    if (scrollTopBtn) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 400) {
                scrollTopBtn.classList.add('visible');
            } else {
                scrollTopBtn.classList.remove('visible');
            }
        });

        scrollTopBtn.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // 4. Form Loading and Validation for Predictions
    const predictForm = document.getElementById('predict-form');
    if (predictForm) {
        predictForm.addEventListener('submit', (e) => {
            // Trigger visual validation
            if (!predictForm.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
                predictForm.classList.add('was-validated');
                showToast('Please fill in all clinical attributes correctly.', 'warning');
                return;
            }

            // If valid, show loading overlay during model prediction pipeline
            if (loader) {
                const loaderText = loader.querySelector('p');
                if (loaderText) {
                    loaderText.innerText = "Analyzing metrics using medical intelligence system...";
                }
                loader.style.opacity = '1';
                loader.style.visibility = 'visible';
            }
        });
    }

    // 5. Toast Notification System
    function showToast(message, type = 'info') {
        let toastContainer = document.querySelector('.toast-container-custom');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container-custom';
            document.body.appendChild(toastContainer);
        }

        const toast = document.createElement('div');
        toast.className = `toast-custom ${type === 'success' ? 'toast-custom-success' : ''}`;
        
        let iconClass = 'bi-info-circle';
        if (type === 'success') iconClass = 'bi-check-circle-fill text-success';
        if (type === 'warning') iconClass = 'bi-exclamation-triangle-fill text-warning';
        
        toast.innerHTML = `
            <i class="bi ${iconClass}"></i>
            <span>${message}</span>
        `;
        
        toastContainer.appendChild(toast);
        
        // Trigger reflow to animate
        setTimeout(() => toast.classList.add('show'), 10);
        
        // Auto remove toast
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 400);
        }, 3500);
    }

    // Expose toast to window so it can be called from template scripts
    window.showToast = showToast;
});
