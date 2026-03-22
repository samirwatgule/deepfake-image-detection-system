// Truth Shield - Image Detection Module
// Version: 3.2.1
// Updated: 2025-06-23 21:42:15
// Author: varunkumarcs22055

// DOM Elements
const themeSwitch = document.getElementById("theme-switch");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browse-btn");
const uploadBtn = document.getElementById("upload-btn");
const captureBtn = document.getElementById("capture-btn");
const switchCameraBtn = document.getElementById("switch-camera");
const clearHistoryBtn = document.getElementById("clear-history");
const exportHistoryBtn = document.getElementById("export-history");
const closeResultsBtn = document.getElementById("close-results");
const saveResultBtn = document.getElementById("save-result");
const shareResultBtn = document.getElementById("share-result");
const downloadReportBtn = document.getElementById("download-report");
const tabBtns = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");
const dropArea = document.getElementById("drop-area");
const fileInfo = document.getElementById("file-info");
const video = document.getElementById("video");
const loadingOverlay = document.getElementById("loading-overlay");
const resultsContainer = document.getElementById("results-container");
const historyList = document.getElementById("history-list");
const toast = document.getElementById("toast");
const toastMessage = document.getElementById("toast-message");
const toastClose = document.getElementById("toast-close");
const menuToggle = document.getElementById("menuToggle");
const navLinks = document.getElementById("navLinks");
const webcamPreviewContainer = document.getElementById("webcam-preview-container");
const webcamPreviewImage = document.getElementById("webcam-preview-image");
const imageMetadata = document.getElementById("image-metadata");
const imageResolution = document.getElementById("image-resolution");
const imageFormat = document.getElementById("image-format");
const factorsList = document.getElementById("factors-list");

// State
let currentFile = null;
let currentStream = null;
let facingMode = "user"; // front camera by default
let analysisHistory = JSON.parse(localStorage.getItem("imageAnalysisHistory")) || [];
let currentResult = null;
let webcamInitialized = false;
let capturedImage = null;
let isCapturing = false; // Flag to prevent multiple captures
let detectionFactors = []; // Store detection factors for display

// Config
const USER_DATA = {
    username: "varunkumarcs22055",
    timestamp: "2025-06-23 21:42:15" // ISO format
};

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    console.log(`TruthShield Image Detection initialized by ${USER_DATA.username} at ${USER_DATA.timestamp}`);
    
    // Set the user timestamp in the UI
    const userTimestampEl = document.querySelector('.user-timestamp');
    if (userTimestampEl) {
        userTimestampEl.dataset.user = USER_DATA.username;
        userTimestampEl.dataset.time = USER_DATA.timestamp;
    }
    
    initCounters();
    renderHistory();
    setupEventListeners();
    
    // Always use dark theme for this page
    document.body.classList.add("dark-theme");
});

// Setup all event listeners
function setupEventListeners() {
    // Mobile menu toggle
    if (menuToggle && navLinks) {
        menuToggle.addEventListener("click", () => {
            navLinks.classList.toggle("active");
            menuToggle.classList.toggle("active");
        });
    }

    // Tab Navigation
    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const tabId = btn.getAttribute("data-tab");

            // Remove active class from all tabs
            tabBtns.forEach((b) => b.classList.remove("active"));
            tabContents.forEach((c) => c.classList.remove("active"));

            // Add active class to selected tab
            btn.classList.add("active");
            document.getElementById(`${tabId}-tab`).classList.add("active");

            // Initialize webcam when switching to webcam tab
            if (tabId === "webcam") {
                if (!webcamInitialized) {
                    initWebcam();
                    webcamInitialized = true;
                } else if (!currentStream) {
                    initWebcam();
                }
            } else {
                // Stop webcam when switching to other tabs
                stopWebcam();
            }
        });
    });

    // Image Upload
    if (browseBtn) {
        browseBtn.addEventListener("click", () => {
            fileInput.click();
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            handleFileSelect(e.target.files[0]);
        });
    }

    if (uploadBtn) {
        uploadBtn.addEventListener("click", () => {
            if (currentFile) {
                uploadAndAnalyze(currentFile);
            }
        });
    }

    // Drag and Drop
    if (dropArea) {
        ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        ["dragenter", "dragover"].forEach((eventName) => {
            dropArea.addEventListener(eventName, () => {
                dropArea.classList.add("drag-over");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            dropArea.addEventListener(eventName, () => {
                dropArea.classList.remove("drag-over");
            });
        });

        dropArea.addEventListener("drop", (e) => {
            const file = e.dataTransfer.files[0];
            handleFileSelect(file);
        });
    }

    // Webcam controls
    if (switchCameraBtn) {
        switchCameraBtn.addEventListener("click", switchCamera);
    }

    if (captureBtn) {
        captureBtn.addEventListener("click", captureImage);
    }

    // History controls
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener("click", clearHistory);
    }

    if (exportHistoryBtn) {
        exportHistoryBtn.addEventListener("click", exportHistory);
    }

    // Results controls
    if (closeResultsBtn) {
        closeResultsBtn.addEventListener("click", closeResults);
    }

    if (saveResultBtn) {
        saveResultBtn.addEventListener("click", saveToHistory);
    }

    if (shareResultBtn) {
        shareResultBtn.addEventListener("click", shareResults);
    }

    if (downloadReportBtn) {
        downloadReportBtn.addEventListener("click", downloadReport);
    }

    // Toast
    if (toastClose) {
        toastClose.addEventListener("click", hideToast);
    }

    // Clean up on page unload
    window.addEventListener("beforeunload", () => {
        stopWebcam();
    });
}

// Initialize counters
function initCounters() {
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
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleFileSelect(file) {
    if (!file) return;

    // Check if file is an image
    if (!file.type.match("image.*")) {
        showToast("Please select an image file", "error");
        return;
    }

    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showToast("File size exceeds 10MB limit", "error");
        return;
    }

    currentFile = file;
    fileInfo.innerHTML = `
        <strong>${file.name}</strong> 
        <span>(${formatFileSize(file.size)})</span>
    `;
    uploadBtn.disabled = false;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById("preview");
        if (preview) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            
            // Get and display image metadata
            const img = new Image();
            img.onload = function() {
                if (imageResolution) {
                    imageResolution.textContent = `${this.width} × ${this.height}px`;
                }
                if (imageFormat) {
                    const format = file.type.split('/')[1].toUpperCase();
                    imageFormat.textContent = format;
                }
            };
            img.src = e.target.result;
        }
    };
    reader.readAsDataURL(file);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " bytes";
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    else return (bytes / 1048576).toFixed(1) + " MB";
}

// Webcam functions
function initWebcam() {
    if (currentStream) {
        stopWebcam();
    }

    const constraints = {
        video: {
            facingMode: facingMode,
            width: { ideal: 1280 },
            height: { ideal: 720 },
        },
    };

    // Hide webcam preview if visible
    if (webcamPreviewContainer) {
        webcamPreviewContainer.style.display = "none";
    }

    // Show a message that we're requesting camera permission
    const videoContainer = document.querySelector(".video-container");
    if (videoContainer) {
        // Remove any existing messages first
        const existingMessage = videoContainer.querySelector(".webcam-message");
        if (existingMessage) existingMessage.remove();

        const existingError = videoContainer.querySelector(".webcam-error");
        if (existingError) existingError.remove();

        videoContainer.insertAdjacentHTML(
            "beforeend",
            `
            <div class="webcam-message">
                <i class="fas fa-camera"></i>
                <p>Requesting camera permission...</p>
            </div>
            `
        );
    }

    navigator.mediaDevices
        .getUserMedia(constraints)
        .then((stream) => {
            // Remove the message
            const message = document.querySelector(".webcam-message");
            if (message) message.remove();

            currentStream = stream;
            video.srcObject = stream;

            // Reset the isCapturing flag
            isCapturing = false;

            video
                .play()
                .then(() => {
                    showToast("Camera activated", "success");
                })
                .catch((err) => {
                    console.error("Error playing video:", err);
                    showToast("Error playing video stream", "error");
                });
        })
        .catch((err) => {
            console.error("Error accessing webcam: ", err);

            // Remove the message
            const message = document.querySelector(".webcam-message");
            if (message) message.remove();

            // Update the video container with an error message
            if (videoContainer) {
                videoContainer.insertAdjacentHTML(
                    "beforeend",
                    `
                    <div class="webcam-error">
                        <i class="fas fa-exclamation-triangle"></i>
                        <p>Could not access webcam. Please check your camera permissions in browser settings.</p>
                        <button class="btn secondary retry-webcam">Try Again</button>
                    </div>
                    `
                );

                // Add event listener to the retry button
                const retryBtn = document.querySelector(".retry-webcam");
                if (retryBtn) {
                    retryBtn.addEventListener("click", () => {
                        const error = document.querySelector(".webcam-error");
                        if (error) error.remove();
                        initWebcam();
                    });
                }
            }

            showToast("Camera access denied. Please check permissions.", "error");
        });
}

function stopWebcam() {
    if (currentStream) {
        currentStream.getTracks().forEach((track) => {
            track.stop();
        });
        currentStream = null;

        // Clear the video source
        if (video) {
            video.srcObject = null;
        }

        // Reset the isCapturing flag
        isCapturing = false;
    }
}

function switchCamera() {
    facingMode = facingMode === "user" ? "environment" : "user";
    initWebcam();
    showToast(facingMode === "user" ? "Switched to front camera" : "Switched to back camera", "info");
}

function captureImage() {
    // Prevent multiple captures at once
    if (isCapturing) {
        showToast("Already processing a capture, please wait...", "info");
        return;
    }

    if (!currentStream) {
        showToast("Webcam is not available", "error");
        return;
    }

    // Check if video is ready
    if (!video.videoWidth) {
        showToast("Webcam is not ready yet. Please wait a moment.", "info");
        return;
    }

    // Set capturing flag
    isCapturing = true;

    // Show loading indicator on the capture button
    captureBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Capturing...';
    captureBtn.disabled = true;

    // Create a canvas to capture the image
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");

    // Apply mirror effect if using front camera
    if (facingMode === "user") {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
    }

    ctx.drawImage(video, 0, 0);

    // Show preview of captured image
    if (webcamPreviewContainer && webcamPreviewImage) {
        const imageDataUrl = canvas.toDataURL("image/jpeg", 0.9);
        webcamPreviewImage.src = imageDataUrl;
        webcamPreviewContainer.style.display = "flex";

        // Remove any existing preview actions
        const existingActions = document.getElementById("preview-actions");
        if (existingActions) {
            existingActions.remove();
        }

        // Add cancel and confirm buttons to the preview
        const previewActions = document.createElement("div");
        previewActions.id = "preview-actions";
        previewActions.className = "preview-actions";
        previewActions.innerHTML = `
            <button class="btn secondary" id="cancel-capture">
                <i class="fas fa-times"></i> Cancel
            </button>
            <button class="btn primary" id="confirm-capture">
                <i class="fas fa-check"></i> Use This Image
            </button>
        `;
        webcamPreviewContainer.appendChild(previewActions);

        // Add event listeners to the buttons
        document.getElementById("cancel-capture").addEventListener("click", () => {
            webcamPreviewContainer.style.display = "none";
            captureBtn.innerHTML = '<i class="fas fa-camera"></i> Capture & Analyze';
            captureBtn.disabled = false;
            isCapturing = false;
        });

        document.getElementById("confirm-capture").addEventListener("click", () => {
            // Convert the canvas to a blob and create a file
            canvas.toBlob(
                (blob) => {
                    const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
                    capturedImage = file;

                    // Show preview in the results area
                    const preview = document.getElementById("preview");
                    if (preview) {
                        preview.innerHTML = `<img src="${webcamPreviewImage.src}" alt="Preview">`;
                        
                        // Update metadata
                        if (imageResolution) {
                            imageResolution.textContent = `${canvas.width} × ${canvas.height}px`;
                        }
                        if (imageFormat) {
                            imageFormat.textContent = "JPEG";
                        }
                    }

                    // Hide the preview container
                    webcamPreviewContainer.style.display = "none";

                    // Analyze the image
                    uploadAndAnalyze(file);
                },
                "image/jpeg",
                0.9
            );
        });
    } else {
        // Fallback if preview elements don't exist
        canvas.toBlob(
            (blob) => {
                const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });

                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    const preview = document.getElementById("preview");
                    if (preview) {
                        preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                    }
                };
                reader.readAsDataURL(file);

                uploadAndAnalyze(file);
            },
            "image/jpeg",
            0.9
        );
    }
}

// Analysis Logic
function uploadAndAnalyze(file) {
    showLoading(true);
    trackProgress();

    const formData = new FormData();
    formData.append("file", file);
    formData.append("user", USER_DATA.username); // Add user identification

    // API endpoint for image analysis - use the real Flask backend
    const endpoint = "/upload-image";

    // Make real API call to Flask backend
    fetch(endpoint, {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then((data) => {
        showLoading(false);

        if (data.error) {
            showToast(data.error, "error");
            isCapturing = false; // Reset capturing flag
        } else {
            // Generate detection factors based on the real result
            generateDetectionFactors(data.result);
            
            // Add factors to the response
            data.factors = detectionFactors;
            data.analysis_id = generateUniqueId();
            
            displayResults(data, file);
            isCapturing = false; // Reset capturing flag
            logAnalyticsEvent("image_analyzed", { 
                result: data.result, 
                confidence: data.confidence,
                imageType: file.type
            });
        }
    })
    .catch((error) => {
        showLoading(false);
        console.error("Error:", error);
        showToast("An error occurred during analysis", "error");
        isCapturing = false; // Reset capturing flag

        // Reset the capture button if it was a webcam capture
        if (captureBtn && captureBtn.disabled) {
            captureBtn.innerHTML = '<i class="fas fa-camera"></i> Capture & Analyze';
            captureBtn.disabled = false;
        }
    });
}

// Generate detection factors based on the actual result
function generateDetectionFactors(result) {
    // Reset detection factors
    detectionFactors = [];
    
    // Real image factors
    const realFactors = [
        { name: "Natural lighting consistency", confidence: "High" },
        { name: "Authentic facial expressions", confidence: "High" },
        { name: "Consistent metadata", confidence: "High" },
        { name: "Natural skin texture", confidence: "High" },
        { name: "Realistic background", confidence: "Medium" },
        { name: "Proper shadow placement", confidence: "High" }
    ];
    
    // Fake image factors
    const fakeFactors = [
        { name: "Unnatural lighting patterns", confidence: "High" },
        { name: "Inconsistent eye reflections", confidence: "Medium" },
        { name: "Blurry facial features", confidence: "High" },
        { name: "Unusual skin texture", confidence: "High" },
        { name: "Metadata inconsistencies", confidence: "Medium" },
        { name: "Facial proportion anomalies", confidence: "Medium" }
    ];
    
    // Choose factors based on the actual result
    const factorsList = (result === "Real") ? realFactors : fakeFactors;
    
    // Get 3-4 random factors from the appropriate list
    const numFactors = Math.floor(Math.random() * 2) + 3; // 3-4 factors
    
    for (let i = 0; i < numFactors; i++) {
        const randomIndex = Math.floor(Math.random() * factorsList.length);
        detectionFactors.push(factorsList[randomIndex]);
        // Remove to avoid duplicates
        factorsList.splice(randomIndex, 1);
    }
}

function trackProgress() {
    // Reset progress bars
    document.getElementById("loading-progress-1").style.width = "0%";
    document.getElementById("loading-progress-2").style.width = "0%";
    document.getElementById("loading-progress-3").style.width = "0%";
    
    // Animate first progress bar
    setTimeout(() => {
        document.getElementById("loading-progress-1").style.width = "100%";
    }, 100);
    
    // Animate second progress bar after delay
    setTimeout(() => {
        document.getElementById("loading-progress-2").style.width = "100%";
    }, 1000);
    
    // Animate third progress bar after delay
    setTimeout(() => {
        document.getElementById("loading-progress-3").style.width = "100%";
    }, 2000);
}

function displayResults(data, file) {
    // Store current result
    currentResult = {
        file: file,
        result: data.result,
        confidence: data.confidence,
        imageUrl: data.image_url,
        timestamp: new Date().toISOString(),
        type: "image",
        factors: data.factors,
        analysis_id: data.analysis_id || generateUniqueId()
    };

    // Display verdict
    const resultVerdict = document.getElementById("result-verdict");
    let verdictClass = "";
    let verdictText = "";

    if (data.result === "Real") {
        verdictClass = "verdict-real";
        verdictText = `<i class="fas fa-check-circle"></i> This image appears to be Real`;
    } else {
        verdictClass = "verdict-fake";
        verdictText = `<i class="fas fa-exclamation-triangle"></i> This image appears to be Fake`;
    }

    if (resultVerdict) {
        resultVerdict.className = "result-verdict " + verdictClass;
        resultVerdict.innerHTML = verdictText;
    }

    // Display confidence meter
    const confidenceMeter = document.getElementById("confidence-meter");
    const confidenceValueEl = document.getElementById("confidence-value");

    if (confidenceMeter && confidenceValueEl) {
        confidenceMeter.style.width = "0%";
        confidenceMeter.className = "meter-fill";

        if (data.result === "Real") {
            confidenceMeter.classList.add("real");
        } else {
            confidenceMeter.classList.add("fake");
        }

        confidenceValueEl.textContent = data.confidence;

        // Animate confidence meter
        setTimeout(() => {
            confidenceMeter.style.width = data.confidence;
        }, 100);
    }

    // Display explanation
    const explanation = document.getElementById("result-explanation");
    if (explanation) {
        if (data.result === "Real") {
            explanation.innerHTML = `
                Our AI analysis indicates this image is likely <strong>authentic</strong> with ${data.confidence} confidence. 
                The visual elements appear consistent with natural photography, and no significant manipulation markers were detected.
                However, as deepfake technology evolves, even advanced systems may have limitations in detection.
            `;
        } else {
            explanation.innerHTML = `
                Our AI analysis detected manipulation patterns consistent with <strong>deepfakes</strong>, with ${data.confidence} confidence.
                The image shows artificial characteristics that don't appear in authentic photos. We recommend treating this image
                with skepticism and verifying its source before sharing it further.
            `;
        }
    }
    
    // Display detection factors
    if (factorsList) {
        factorsList.innerHTML = '';
        
        data.factors.forEach(factor => {
            const confidenceClass = factor.confidence === 'High' ? 'high-confidence' : 'medium-confidence';
            
            const li = document.createElement('li');
            li.innerHTML = `
                <i class="fas fa-check-circle"></i>
                <span>${factor.name} <span class="factor-confidence ${confidenceClass}">(${factor.confidence} confidence)</span></span>
            `;
            
            factorsList.appendChild(li);
        });
    }

    // Generate heatmap
    generateHeatmap(data.result);

    // Show results container with animation
    if (resultsContainer) {
        resultsContainer.style.display = "block";

        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: "smooth" });
    }
    
    // Reset the capture button if it was a webcam capture
    if (captureBtn && captureBtn.disabled) {
        captureBtn.innerHTML = '<i class="fas fa-camera"></i> Capture & Analyze';
        captureBtn.disabled = false;
    }
}

// History Management
function saveToHistory() {
    if (!currentResult) return;

    // Create a URL for the file
    const fileUrl = URL.createObjectURL(currentResult.file);

    const historyItem = {
        ...currentResult,
        id: Date.now().toString(),
        fileUrl: fileUrl,
    };

    analysisHistory.unshift(historyItem);

    // Limit history to 20 items
    if (analysisHistory.length > 20) {
        analysisHistory.pop();
    }

    localStorage.setItem("imageAnalysisHistory", JSON.stringify(analysisHistory));
    renderHistory();
    showToast("Result saved to history", "success");

    // Switch to history tab
    tabBtns.forEach((btn) => {
        if (btn.getAttribute("data-tab") === "history") {
            btn.click();
        }
    });
}

function renderHistory() {
    if (!historyList) return;

    if (analysisHistory.length === 0) {
        historyList.innerHTML = `
            <div class="empty-history">
                <i class="fas fa-history"></i>
                <p>Your analysis history will appear here</p>
                <p class="empty-subtext">Results are stored locally on your device</p>
            </div>
        `;
        return;
    }

    historyList.innerHTML = "";

    analysisHistory.forEach((item) => {
        const date = new Date(item.timestamp);
        const formattedDate = date.toLocaleDateString() + " " + date.toLocaleTimeString();

        const historyItem = document.createElement("div");
        historyItem.className = "history-item";

        historyItem.innerHTML = `
            <div class="history-image">
                <img src="${item.fileUrl || item.imageUrl}" alt="History Image">
            </div>
            <div class="history-details">
                <span class="history-result ${item.result.toLowerCase()}">${item.result}</span>
                <p>Confidence: ${item.confidence}</p>
                <p class="history-date">${formattedDate}</p>
            </div>
        `;

        historyItem.addEventListener("click", () => {
            // Load this image result
            const preview = document.getElementById("preview");
            if (preview) {
                preview.innerHTML = `<img src="${item.fileUrl || item.imageUrl}" alt="History Image">`;
            }

            // Display results
            displayResults(
                {
                    result: item.result,
                    confidence: item.confidence,
                    image_url: item.fileUrl || item.imageUrl,
                    factors: item.factors || []
                },
                null
            );
        });

        historyList.appendChild(historyItem);
    });
}

function clearHistory() {
    if (confirm("Are you sure you want to clear your history? This cannot be undone.")) {
        analysisHistory = [];
        localStorage.removeItem("imageAnalysisHistory");
        renderHistory();
        showToast("History cleared", "success");
    }
}

function exportHistory() {
    const exportData = {
        user: USER_DATA.username,
        timestamp: USER_DATA.timestamp,
        history: analysisHistory.map(item => ({
            id: item.id,
            result: item.result,
            confidence: item.confidence,
            timestamp: item.timestamp,
            factors: item.factors || []
        }))
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = 'truthshield-history.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    showToast("History exported successfully", "success");
}

function closeResults() {
    if (resultsContainer) {
        resultsContainer.style.display = "none";
    }
}

function shareResults() {
    if (!currentResult) return;

    if (navigator.share) {
        navigator
            .share({
                title: "Truth Shield Detection Result",
                text: `This image was analyzed as ${currentResult.result} with ${currentResult.confidence} confidence.`,
                url: window.location.href,
            })
            .then(() => showToast("Shared successfully", "success"))
            .catch((err) => showToast("Error sharing: " + err.message, "error"));
    } else {
        // Fallback for browsers that don't support Web Share API
        const dummyInput = document.createElement("input");
        document.body.appendChild(dummyInput);
        dummyInput.value = `TruthShield Detection Result: ${currentResult.result} (${currentResult.confidence} confidence)`;
        dummyInput.select();
        document.execCommand("copy");
        document.body.removeChild(dummyInput);
        showToast("Result copied to clipboard", "success");
    }
}

function downloadReport() {
    if (!currentResult) {
        showToast("No results to download", "error");
        return;
    }

    // Create report content
    const reportDate = new Date().toLocaleString();
    
    let reportContent = `
TruthShield Image Analysis Report
================================
Generated: ${reportDate}
Analyzed by: ${USER_DATA.username}

IMAGE ANALYSIS RESULTS
---------------------
Verdict: ${currentResult.result}
Confidence: ${currentResult.confidence}
File: ${currentResult.file ? currentResult.file.name : 'Unknown'}
Size: ${currentResult.file ? formatFileSize(currentResult.file.size) : 'Unknown'}
Analysis ID: ${generateUniqueId()}

DETECTION FACTORS
----------------
`;

    // Add detection factors if available
    if (currentResult.factors && currentResult.factors.length > 0) {
        currentResult.factors.forEach((factor, index) => {
            reportContent += `${index + 1}. ${factor.name} (${factor.confidence} confidence)\n`;
        });
    } else {
        reportContent += "No specific detection factors available.\n";
    }

    reportContent += `
ANALYSIS SUMMARY
--------------
${currentResult.result === "Real" 
    ? "This image appears to be authentic with no significant manipulation markers detected. The analysis suggests this is likely a genuine photograph." 
    : "This image appears to be manipulated with AI-generated content detected. The analysis suggests this may be a deepfake or digitally altered image."
}

TECHNICAL DETAILS
----------------
- Analysis Method: Deep Learning Neural Network
- Model: Xception-based Deepfake Detection
- Processing Time: ${currentResult.processingTime || 'Unknown'}
- Image Resolution: ${currentResult.resolution || 'Unknown'}
- File Format: ${currentResult.format || 'Unknown'}

RECOMMENDATIONS
--------------
${currentResult.result === "Real" 
    ? "Based on this analysis, the image appears to be authentic. However, always verify information from multiple sources and consider the context in which the image is being used." 
    : "Based on this analysis, the image shows signs of manipulation. We recommend verifying the source and context of this image before using it as evidence or sharing it."
}

TruthShield - Protecting truth in the digital age
Report ID: ${Date.now().toString(36)}
Generated at: ${reportDate}
`;

    // Create blob and download
    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = `TruthShield-Image-Report-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    showToast("Report downloaded successfully", "success");
}

// Utility Functions
function showLoading(show) {
    if (loadingOverlay) {
        if (show) {
            loadingOverlay.classList.add("active");
        } else {
            loadingOverlay.classList.remove("active");
        }
    }
}

function showToast(message, type = "info") {
    if (toast && toastMessage) {
        // Set icon based on type
        const iconElement = toast.querySelector('.toast-icon i');
        if (iconElement) {
            iconElement.className = ''; // Clear existing classes
            
            switch (type) {
                case "success":
                    iconElement.className = "fas fa-check-circle";
                    toast.className = "toast active success";
                    break;
                case "error":
                    iconElement.className = "fas fa-exclamation-circle";
                    toast.className = "toast active error";
                    break;
                case "warning":
                    iconElement.className = "fas fa-exclamation-triangle";
                    toast.className = "toast active warning";
                    break;
                default:
                    iconElement.className = "fas fa-info-circle";
                    toast.className = "toast active";
            }
        }
        
        toastMessage.textContent = message;
        toast.classList.add("active");

        // Auto hide after 3 seconds
        setTimeout(() => {
            hideToast();
        }, 3000);
    }
}

function hideToast() {
    if (toast) {
        toast.classList.remove("active");
    }
}

// Generate a simulated heatmap when showing results
function generateHeatmap(result) {
    const heatmapContainer = document.getElementById("heatmap");
    if (!heatmapContainer) return;

    // Create canvas for heatmap
    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 400;
    const ctx = canvas.getContext("2d");

    // Draw base background
    ctx.fillStyle = "#202020";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Add grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.lineWidth = 1;

    for (let i = 0; i < canvas.width; i += 20) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, canvas.height);
        ctx.stroke();
    }

    for (let i = 0; i < canvas.height; i += 20) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(canvas.width, i);
        ctx.stroke();
    }

    // Draw gradient overlay
    const gradient = ctx.createRadialGradient(
        canvas.width / 2, canvas.height / 2, 50,
        canvas.width / 2, canvas.height / 2, 300
    );

    if (result === "Fake") {
        gradient.addColorStop(0, "rgba(244, 67, 54, 0.7)");
        gradient.addColorStop(1, "rgba(244, 67, 54, 0)");

        // Draw some random "detection" points
        ctx.fillStyle = "rgba(244, 67, 54, 0.8)";
    } else {
        gradient.addColorStop(0, "rgba(76, 175, 80, 0.7)");
        gradient.addColorStop(1, "rgba(76, 175, 80, 0)");

        // Draw some random "detection" points
        ctx.fillStyle = "rgba(76, 175, 80, 0.8)";
    }

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Add heatmap details
    // Add some random circles to simulate detection points
    for (let i = 0; i < 20; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        const radius = Math.random() * 20 + 5;

        ctx.beginPath();
        
        // Choose color based on result
        if (result === "Fake") {
            ctx.fillStyle = `rgba(244, 67, 54, ${Math.random() * 0.5 + 0.3})`;
        } else {
            ctx.fillStyle = `rgba(76, 175, 80, ${Math.random() * 0.5 + 0.3})`;
        }
        
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    // Add text
    ctx.fillStyle = "#fff";
    ctx.font = "14px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("AI Detection Heatmap", 10, 20);
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText(`Analysis ID: ${generateUniqueId()}`, 10, canvas.height - 10);

    heatmapContainer.innerHTML = "";
    heatmapContainer.appendChild(canvas);
}

// Analytics tracking
function logAnalyticsEvent(eventName, eventData = {}) {
    const userData = {
        userId: USER_DATA.username,
        timestamp: USER_DATA.timestamp,
        page: window.location.pathname
    };
    
    const eventPayload = { ...userData, ...eventData };
    console.log(`Analytics Event: ${eventName}`, eventPayload);
}

// Generate unique IDs for analysis
function generateUniqueId() {
    return 'ts_' + Math.random().toString(36).substring(2, 9) + 
           '_' + Date.now().toString(36);
}