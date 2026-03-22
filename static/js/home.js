// Initialize AOS (Animate On Scroll)
document.addEventListener('DOMContentLoaded', function() {
    // Set user and timestamp
    const currentUser = 'varunkumarcs22055';
    const lastUpdated = '2025-06-23 20:29:21';
    console.log(`Truth Shield initialized by ${currentUser} at ${lastUpdated}`);
    
    AOS.init({
        duration: 1000,
        once: true,
        offset: 100,
        easing: 'ease-out'
    });

    // Mobile Menu Toggle
    const menuToggle = document.getElementById('menuToggle');
    const navLinks = document.getElementById('navLinks');

    if (menuToggle && navLinks) {
        menuToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            menuToggle.classList.toggle('active');
        });
    }

    // Counter Animation
    const counters = document.querySelectorAll('.counter');
    const speed = 200;

    counters.forEach(counter => {
        const animate = () => {
            const value = +counter.getAttribute('data-target');
            const data = +counter.innerText;
            
            const time = value / speed;
            if (data < value) {
                counter.innerText = Math.ceil(data + time);
                setTimeout(animate, 1);
            } else {
                counter.innerText = value;
            }
        }
        
        animate();
    });

    // Create Particle Background
    createParticles();

    // Initialize Chart.js if chart container exists
    const chartContainer = document.getElementById('detectionChart');
    if (chartContainer) {
        const ctx = chartContainer.getContext('2d');
        
        const gradientFill = ctx.createLinearGradient(0, 0, 0, 400);
        gradientFill.addColorStop(0, 'rgba(0, 198, 255, 0.3)');
        gradientFill.addColorStop(0.8, 'rgba(0, 198, 255, 0.05)');
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
                datasets: [{
                    label: 'Deepfakes Detected',
                    data: [65, 78, 90, 85, 99, 112, 120],
                    borderColor: '#00c6ff',
                    borderWidth: 3,
                    pointBackgroundColor: '#9d4edd',
                    pointBorderColor: '#9d4edd',
                    pointRadius: 5,
                    tension: 0.4,
                    fill: true,
                    backgroundColor: gradientFill
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(32, 32, 32, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: 'rgba(0, 198, 255, 0.3)',
                        borderWidth: 1,
                        displayColors: false,
                        padding: 12,
                        callbacks: {
                            label: function(context) {
                                return `Detected: ${context.parsed.y} deepfakes`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false,
                            drawBorder: false
                        },
                        ticks: {
                            color: '#808080'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#808080'
                        }
                    }
                }
            }
        });
    }

    // Pricing Toggle
    const billingToggle = document.getElementById('billingToggle');
    const pricingCards = document.querySelectorAll('.pricing-card .amount');
    
    if (billingToggle && pricingCards.length > 0) {
        const monthlyPrices = ['49', '149', '499'];
        const annualPrices = ['39', '119', '399'];
        
        billingToggle.addEventListener('change', function() {
            const isAnnual = this.checked;
            const priceArray = isAnnual ? annualPrices : monthlyPrices;
            
            pricingCards.forEach((card, index) => {
                card.textContent = priceArray[index];
                
                // Animate price change
                card.style.animation = 'none';
                setTimeout(() => {
                    card.style.animation = 'glow 2.5s ease-in-out infinite alternate';
                }, 10);
            });
            
            document.querySelectorAll('.toggle-option').forEach(option => {
                option.classList.toggle('active');
            });
        });
    }

    // Video Player
    const videoPlaceholder = document.querySelector('.video-placeholder');
    if (videoPlaceholder) {
        videoPlaceholder.addEventListener('click', function() {
            const videoContainer = this.parentElement;
            const video = document.createElement('iframe');
            video.src = "https://www.youtube.com/embed/dQw4w9WgXcQ?autoplay=1"; // Replace with your actual demo video
            video.width = "100%";
            video.height = "100%";
            video.frameBorder = "0";
            video.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
            video.allowFullscreen = true;
            
            videoContainer.innerHTML = '';
            videoContainer.appendChild(video);
        });
    }

    // Testimonial Carousel
    const testimonialDots = document.querySelectorAll('.testimonial-dots .dot');
    const testimonialCards = document.querySelectorAll('.testimonial-card');
    const prevBtn = document.querySelector('.testimonial-nav.prev');
    const nextBtn = document.querySelector('.testimonial-nav.next');
    
    if (testimonialDots.length > 0 && testimonialCards.length > 0) {
        let currentSlide = 0;
        
        const goToSlide = (slideIndex) => {
            currentSlide = slideIndex;
            
            testimonialCards.forEach((card, idx) => {
                if (idx === slideIndex) {
                    card.style.opacity = '1';
                    card.style.transform = 'translateX(0)';
                } else {
                    card.style.opacity = '0';
                    card.style.transform = idx < slideIndex ? 'translateX(-10%)' : 'translateX(10%)';
                }
            });
            
            testimonialDots.forEach((dot, idx) => {
                dot.classList.toggle('active', idx === slideIndex);
            });
        };
        
        testimonialDots.forEach((dot, idx) => {
            dot.addEventListener('click', () => goToSlide(idx));
        });
        
        if (prevBtn && nextBtn) {
            prevBtn.addEventListener('click', () => {
                const newIndex = currentSlide === 0 ? testimonialCards.length - 1 : currentSlide - 1;
                goToSlide(newIndex);
            });
            
            nextBtn.addEventListener('click', () => {
                const newIndex = currentSlide === testimonialCards.length - 1 ? 0 : currentSlide + 1;
                goToSlide(newIndex);
            });
        }
        
        // Auto slide every 5 seconds
        setInterval(() => {
            const newIndex = currentSlide === testimonialCards.length - 1 ? 0 : currentSlide + 1;
            goToSlide(newIndex);
        }, 5000);
    }

    // Cookie Consent
    const cookieConsent = document.getElementById('cookieConsent');
    const acceptCookies = document.querySelector('.cookie-button.accept');
    const customizeCookies = document.querySelector('.cookie-button.customize');
    
    if (cookieConsent && acceptCookies) {
        const hasConsent = localStorage.getItem('cookieConsent');
        
        if (!hasConsent) {
            setTimeout(() => {
                cookieConsent.style.display = 'block';
            }, 2000);
        }
        
        acceptCookies.addEventListener('click', () => {
            localStorage.setItem('cookieConsent', 'true');
            localStorage.setItem('cookieConsentDate', lastUpdated);
            cookieConsent.style.animation = 'slide-down 0.5s forwards';
            setTimeout(() => {
                cookieConsent.style.display = 'none';
            }, 500);
        });
        
        if (customizeCookies) {
            customizeCookies.addEventListener('click', () => {
                window.location.href = 'cookies.html';
            });
        }
    }

    // Scan animation for hero image
    animateScan();

    // Add parallax effect
    window.addEventListener('scroll', parallaxEffect);

    // Add event listeners for service boxes
    initServiceBoxes();

    // Initialize verification badges
    initVerificationBadges();
});

// Particle Background
function createParticles() {
    const particles = document.getElementById('particles');
    if (!particles) return;
    
    const particleCount = 40;
    
    for (let i = 0; i < particleCount; i++) {
        const size = Math.random() * 4 + 1;
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${Math.random() * 100}vw`;
        particle.style.top = `${Math.random() * 100}vh`;
        particle.style.opacity = Math.random() * 0.5;
        particle.style.background = Math.random() > 0.5 ? '#00c6ff' : '#9d4edd';
        particle.style.boxShadow = `0 0 ${Math.random() * 10 + 5}px ${particle.style.background}`;
        particle.style.animation = `float ${Math.random() * 10 + 10}s linear infinite`;
        particle.style.borderRadius = '50%';
        particles.appendChild(particle);
    }
}

// Animate the scan effect on the hero image
function animateScan() {
    const scanText = document.querySelector('.scan-text');
    const scanResult = document.querySelector('.scan-result');
    
    if (!scanText || !scanResult) return;
    
    const phrases = [
        "Analyzing facial features...",
        "Checking image metadata...",
        "Examining light consistency...",
        "Verifying pixel patterns...",
        "Running neural analysis..."
    ];
    
    let currentIndex = 0;
    
    setInterval(() => {
        currentIndex = (currentIndex + 1) % phrases.length;
        scanText.textContent = phrases[currentIndex];
    }, 3000);
}

// Parallax effect on scroll
function parallaxEffect() {
    const scrolled = window.scrollY;
    
    // Hero section parallax
    const heroContent = document.querySelector('.hero-content');
    const heroImage = document.querySelector('.hero-image');
    
    if (heroContent && heroImage && scrolled < 1000) {
        heroContent.style.transform = `translateY(${scrolled * 0.1}px)`;
        heroImage.style.transform = `translateY(${scrolled * 0.15}px)`;
    }
    
    // Navbar background opacity
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        const opacity = Math.min(0.95, 0.85 + (scrolled * 0.0001));
        navbar.style.background = `rgba(18, 18, 18, ${opacity})`;
        navbar.style.boxShadow = scrolled > 50 ? '0 2px 20px rgba(0, 0, 0, 0.3)' : '0 2px 15px rgba(0, 0, 0, 0.3)';
    }
    
    // Particles parallax
    const particles = document.querySelectorAll('.particle');
    particles.forEach(particle => {
        const speed = parseFloat(particle.style.width) * 0.02;
        particle.style.transform = `translateY(${scrolled * speed}px)`;
    });
}

// Service box interactions
function initServiceBoxes() {
    const serviceBoxes = document.querySelectorAll('.service-box');
    
    serviceBoxes.forEach(box => {
        box.addEventListener('mouseenter', () => {
            const tryNow = box.querySelector('.try-now');
            if (tryNow) {
                tryNow.style.transform = 'translateX(5px)';
            }
        });
        
        box.addEventListener('mouseleave', () => {
            const tryNow = box.querySelector('.try-now');
            if (tryNow) {
                tryNow.style.transform = 'translateX(0)';
            }
        });
    });
}

// Verification badges
function initVerificationBadges() {
    // This would typically fetch data from your backend
    const verificationData = {
        verified: Math.floor(Math.random() * 1000) + 5000,
        last24h: Math.floor(Math.random() * 100) + 300,
        generated: new Date().toISOString()
    };
    
    // Store in localStorage for demo purposes
    localStorage.setItem('verificationStats', JSON.stringify(verificationData));
    
    // Update any badges on the page
    const verifiedCountElements = document.querySelectorAll('.verified-count');
    if (verifiedCountElements.length > 0) {
        verifiedCountElements.forEach(el => {
            el.textContent = verificationData.verified.toLocaleString();
        });
    }
}

// Add keyboard navigation accessibility
document.addEventListener('keydown', function(e) {
    // Escape key closes mobile menu
    if (e.key === 'Escape') {
        const navLinks = document.getElementById('navLinks');
        const menuToggle = document.getElementById('menuToggle');
        
        if (navLinks && navLinks.classList.contains('active')) {
            navLinks.classList.remove('active');
            if (menuToggle) {
                menuToggle.classList.remove('active');
            }
        }
    }
});

// Detect and log analytics data
function logAnalyticsEvent(eventName, eventData = {}) {
    const userData = {
        userId: 'varunkumarcs22055',
        timestamp: '2025-06-23 20:29:21',
        page: window.location.pathname
    };
    
    const eventPayload = { ...userData, ...eventData };
    console.log(`Analytics Event: ${eventName}`, eventPayload);
    
    // In a real implementation, this would send data to your analytics service
    // fetch('/api/analytics', {
    //     method: 'POST',
    //     headers: { 'Content-Type': 'application/json' },
    //     body: JSON.stringify(eventPayload)
    // });
}

// Log page visit on load
window.addEventListener('load', () => {
    logAnalyticsEvent('page_view');
});

// Track service clicks
document.querySelectorAll('.service-box').forEach(box => {
    box.addEventListener('click', () => {
        const serviceName = box.querySelector('h3').textContent;
        logAnalyticsEvent('service_click', { service: serviceName });
    });
});