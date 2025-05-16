document.getElementById("deepfakeForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const originalFile = document.getElementById("originalFile").files[0];
    const testFile = document.getElementById("testFile").files[0];
    const resultDiv = document.getElementById("result");
    const heatmapCanvas = document.getElementById("heatmap");

    // Reset UI
    resultDiv.style.display = "none";
    heatmapCanvas.style.display = "none";
    resultDiv.textContent = "";

    if (!originalFile || !testFile) {
        resultDiv.className = "result error";
        resultDiv.textContent = "Please upload both files.";
        resultDiv.style.display = "block";
        return;
    }

    try {
        // Load face-api models
        await Promise.all([
            faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
            faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
            faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
        ]);

        // Extract descriptors
        const originalDescriptors = originalFile.type.startsWith("video")
            ? await extractVideoFaceDescriptors(originalFile)
            : await extractFaceDescriptors(originalFile);

        const testDescriptors = testFile.type.startsWith("video")
            ? await extractVideoFaceDescriptors(testFile)
            : await extractFaceDescriptors(testFile);

        if (!originalDescriptors.length || !testDescriptors.length) {
            resultDiv.className = "result error";
            resultDiv.textContent = "No faces detected in one or both files.";
            resultDiv.style.display = "block";
            return;
        }

        // Match faces
        const faceMatcher = new faceapi.FaceMatcher(originalDescriptors);
        const matches = testDescriptors.map((descriptor) =>
            faceMatcher.findBestMatch(descriptor)
        );
        const matchPercent =
            (matches.filter((m) => m.label !== "unknown").length / matches.length) *
            100;

        let heatmapColor = "yellow"; // Default color
        let resultMessage = "";

        if (matchPercent === 100) {
            heatmapColor = "yellow";
            resultDiv.className = "result success";
            resultMessage = "No deepfake detected. Similar faces detected.";
        } else if (matchPercent > 0) {
            heatmapColor = "orange";
            resultDiv.className = "result info";
            resultMessage = `Deepfake detected. Similarity: ${matchPercent.toFixed(
                2
            )}%`;
        } else {
            heatmapColor = "red";
            resultDiv.className = "result error";
            resultMessage = "Deepfake detected. No similarity in faces.";
        }

        resultDiv.textContent = resultMessage;
        resultDiv.style.display = "block";

        // Generate heatmap data dynamically based on match percent
        const heatmapData = Array.from({ length: 10 }, (_, x) =>
            Array.from({ length: 10 }, (_, y) => ({
                x,
                y,
                v: heatmapColor === "red" ? 1 : heatmapColor === "orange" ? 0.5 : 0.2,
            }))
        ).flat();

        // Display heatmap
        heatmapCanvas.style.display = "block";
        displayHeatmap(heatmapCanvas, heatmapData);
    } catch (error) {
        resultDiv.className = "result error";
        resultDiv.textContent = `Error: ${error.message}`;
        resultDiv.style.display = "block";
    }
});

// Helper functions for face and video frame extraction
async function extractFaceDescriptors(file) {
    const img = await fileToImage(file);
    const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();
    return detections.map((d) => d.descriptor);
}

async function extractVideoFaceDescriptors(file) {
    const video = await fileToVideo(file);
    const descriptors = [];
    return new Promise((resolve) => {
        video.addEventListener("loadeddata", async () => {
            const canvas = document.createElement("canvas");
            const context = canvas.getContext("2d");
            video.play();
            video.addEventListener("timeupdate", async () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                const detections = await faceapi.detectAllFaces(canvas).withFaceLandmarks().withFaceDescriptors();
                descriptors.push(...detections.map((d) => d.descriptor));

                if (video.currentTime >= video.duration) {
                    video.pause();
                    resolve(descriptors);
                }
            });
        });
    });
}

function fileToImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.src = reader.result;
        };
        reader.readAsDataURL(file);
    });
}

function fileToVideo(file) {
    return new Promise((resolve) => {
        const video = document.createElement("video");
        const reader = new FileReader();
        video.muted = true; // Ensure the video is muted
        video.controls = false;
        reader.onload = () => {
            video.src = reader.result;
            resolve(video);
        };
        reader.readAsDataURL(file);
    });
}

// Heatmap display function
function displayHeatmap(canvas, data) {
    new Chart(canvas, {
        type: "matrix",
        data: {
            datasets: [
                {
                    label: "Heatmap",
                    data: data,
                    backgroundColor(ctx) {
                        const value = ctx.raw.v;
                        if (value < 0.3) return "rgba(144, 238, 144, 0.8)"; // Light green
                        if (value < 0.7) return "rgba(255, 200, 124, 0.8)"; // Light orange
                        return "rgba(255, 102, 102, 0.8)"; // Light red
                    },
                    borderWidth: 1,
                    width(ctx) {
                        const chart = ctx.chart;
                        if (chart.chartArea) {
                            return chart.chartArea.width / 10 - 2;
                        }
                        return 20;
                    },
                    height(ctx) {
                        const chart = ctx.chart;
                        if (chart.chartArea) {
                            return chart.chartArea.height / 10 - 2;
                        }
                        return 20;
                    },
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { type: "linear", position: "bottom" },
                y: { type: "linear" },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });
}
// Global variables
let originalStream = null;
let testStream = null;
let originalImageBlob = null;
let testImageBlob = null;
let modelsLoaded = false;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    // Load models first
    try {
        await loadModels();
        modelsLoaded = true;
        
        // Setup event listeners after models are loaded
        setupEventListeners();
    } catch (error) {
        console.error("Model loading failed:", error);
        showResult("Failed to load face detection models. Please refresh the page.", "error");
    }
});

async function loadModels() {
    // Clear any previous model cache
    faceapi.tf.enableProdMode();
    faceapi.tf.ENV.set('IS_BROWSER', true);
    
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
        faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
    ]);
}

function setupEventListeners() {
    // Original webcam controls
    document.getElementById('startOriginalWebcam').addEventListener('click', () => startWebcam('original'));
    document.getElementById('stopOriginalWebcam').addEventListener('click', () => stopWebcam('original'));
    document.getElementById('captureOriginal').addEventListener('click', () => captureImage('original'));
    
    // Test webcam controls
    document.getElementById('startTestWebcam').addEventListener('click', () => startWebcam('test'));
    document.getElementById('stopTestWebcam').addEventListener('click', () => stopWebcam('test'));
    document.getElementById('captureTest').addEventListener('click', () => captureImage('test'));
    
    // Form submission
    document.getElementById("deepfakeForm").addEventListener("submit", analyzeFiles);
}

async function analyzeFiles(event) {
    event.preventDefault();

    if (!modelsLoaded) {
        showResult("Face detection models are not loaded yet. Please wait.", "error");
        return;
    }

    const originalFile = document.getElementById("originalFile").files[0];
    const testFile = document.getElementById("testFile").files[0];
    const resultDiv = document.getElementById("result");
    const heatmapCanvas = document.getElementById("heatmap");

    // Reset UI
    resultDiv.style.display = "none";
    heatmapCanvas.style.display = "none";
    resultDiv.textContent = "";

    // Use captured images if available
    const { originalFile: finalOriginalFile, testFile: finalTestFile } = getInputFiles();
    
    if (!finalOriginalFile || !finalTestFile) {
        showResult("Please provide both original and test files/images.", "error");
        return;
    }

    try {
        // Extract descriptors
        const originalDescriptors = finalOriginalFile.type.startsWith("video")
            ? await extractVideoFaceDescriptors(finalOriginalFile)
            : await extractFaceDescriptors(finalOriginalFile);

        const testDescriptors = finalTestFile.type.startsWith("video")
            ? await extractVideoFaceDescriptors(finalTestFile)
            : await extractFaceDescriptors(finalTestFile);

        if (!originalDescriptors.length || !testDescriptors.length) {
            showResult("No faces detected in one or both sources.", "error");
            return;
        }

        // Match faces
        const faceMatcher = new faceapi.FaceMatcher(originalDescriptors);
        const matches = testDescriptors.map((descriptor) =>
            faceMatcher.findBestMatch(descriptor)
        );
        const matchPercent =
            (matches.filter((m) => m.label !== "unknown").length / matches.length) *
            100;

        let heatmapColor = "yellow"; // Default color
        let resultMessage = "";

        if (matchPercent === 100) {
            heatmapColor = "yellow";
            resultDiv.className = "result success";
            resultMessage = "No deepfake detected. Similar faces detected.";
        } else if (matchPercent > 0) {
            heatmapColor = "orange";
            resultDiv.className = "result info";
            resultMessage = `Deepfake detected. Similarity: ${matchPercent.toFixed(2)}%`;
        } else {
            heatmapColor = "red";
            resultDiv.className = "result error";
            resultMessage = "Deepfake detected. No similarity in faces.";
        }

        resultDiv.textContent = resultMessage;
        resultDiv.style.display = "block";

        // Generate heatmap data dynamically based on match percent
        const heatmapData = Array.from({ length: 10 }, (_, x) =>
            Array.from({ length: 10 }, (_, y) => ({
                x,
                y,
                v: heatmapColor === "red" ? 1 : heatmapColor === "orange" ? 0.5 : 0.2,
            }))
        ).flat();

        // Display heatmap
        heatmapCanvas.style.display = "block";
        displayHeatmap(heatmapCanvas, heatmapData);
    } catch (error) {
        console.error("Analysis error:", error);
        showResult(`Error during analysis: ${error.message}`, "error");
    }
}

function getInputFiles() {
    let originalFile, testFile;
    
    if (originalImageBlob) {
        originalFile = new File([originalImageBlob], "original.jpg", { type: "image/jpeg" });
    } else {
        originalFile = document.getElementById("originalFile").files[0];
    }
    
    if (testImageBlob) {
        testFile = new File([testImageBlob], "test.jpg", { type: "image/jpeg" });
    } else {
        testFile = document.getElementById("testFile").files[0];
    }
    
    return { originalFile, testFile };
}

// Face and video processing functions
async function extractFaceDescriptors(file) {
    const img = await fileToImage(file);
    const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();
    return detections.map((d) => d.descriptor);
}

async function extractVideoFaceDescriptors(file) {
    const video = await fileToVideo(file);
    const descriptors = [];
    return new Promise((resolve) => {
        video.addEventListener("loadeddata", async () => {
            const canvas = document.createElement("canvas");
            const context = canvas.getContext("2d");
            video.play();
            video.addEventListener("timeupdate", async () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                const detections = await faceapi.detectAllFaces(canvas).withFaceLandmarks().withFaceDescriptors();
                descriptors.push(...detections.map((d) => d.descriptor));

                if (video.currentTime >= video.duration) {
                    video.pause();
                    resolve(descriptors);
                }
            });
        });
    });
}

function fileToImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.src = reader.result;
        };
        reader.readAsDataURL(file);
    });
}

function fileToVideo(file) {
    return new Promise((resolve) => {
        const video = document.createElement("video");
        const reader = new FileReader();
        video.muted = true; // Ensure the video is muted
        video.controls = false;
        reader.onload = () => {
            video.src = reader.result;
            resolve(video);
        };
        reader.readAsDataURL(file);
    });
}

// Heatmap display function
function displayHeatmap(canvas, data) {
    // Clear previous chart if exists
    if (canvas.chart) {
        canvas.chart.destroy();
    }
    
    canvas.chart = new Chart(canvas, {
        type: "matrix",
        data: {
            datasets: [
                {
                    label: "Heatmap",
                    data: data,
                    backgroundColor(ctx) {
                        const value = ctx.raw.v;
                        if (value < 0.3) return "rgba(255, 255, 0, 0.8)"; // Yellow
                        if (value < 0.7) return "rgba(255, 165, 0, 0.8)"; // Orange
                        return "rgba(255, 0, 0, 0.8)"; // Red
                    },
                    borderWidth: 1,
                    width(ctx) {
                        const chart = ctx.chart;
                        if (chart.chartArea) {
                            return chart.chartArea.width / 10 - 2;
                        }
                        return 20;
                    },
                    height(ctx) {
                        const chart = ctx.chart;
                        if (chart.chartArea) {
                            return chart.chartArea.height / 10 - 2;
                        }
                        return 20;
                    },
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { type: "linear", position: "bottom", display: false, min: 0, max: 9 },
                y: { type: "linear", display: false, min: 0, max: 9 },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw.v;
                            if (value < 0.3) return "No deepfake detected";
                            if (value < 0.7) return "Potential deepfake detected";
                            return "Deepfake detected";
                        }
                    }
                }
            },
        },
    });
}

// Webcam functions
async function startWebcam(type) {
    try {
        const video = document.getElementById(`${type}Webcam`);
        const startBtn = document.getElementById(`start${capitalize(type)}Webcam`);
        const stopBtn = document.getElementById(`stop${capitalize(type)}Webcam`);
        const captureBtn = document.getElementById(`capture${capitalize(type)}`);

        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false 
        });

        if (type === 'original') {
            originalStream = stream;
            originalImageBlob = null;
        } else {
            testStream = stream;
            testImageBlob = null;
        }

        video.srcObject = stream;
        video.classList.remove("hidden");
        startBtn.classList.add("hidden");
        stopBtn.classList.remove("hidden");
        captureBtn.classList.remove("hidden");
    } catch (err) {
        console.error("Webcam error:", err);
        showResult(`Webcam error: ${err.message}`, "error");
    }
}

function stopWebcam(type) {
    const video = document.getElementById(`${type}Webcam`);
    const startBtn = document.getElementById(`start${capitalize(type)}Webcam`);
    const stopBtn = document.getElementById(`stop${capitalize(type)}Webcam`);
    const captureBtn = document.getElementById(`capture${capitalize(type)}`);

    const stream = type === 'original' ? originalStream : testStream;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    video.srcObject = null;
    video.classList.add("hidden");
    startBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    captureBtn.classList.add("hidden");

    if (type === 'original') originalStream = null;
    else testStream = null;
}

function captureImage(type) {
    const video = document.getElementById(`${type}Webcam`);
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(blob => {
        if (type === 'original') originalImageBlob = blob;
        else testImageBlob = blob;
        showResult(`${capitalize(type)} image captured!`, "success");
    }, 'image/jpeg', 0.9);
}

// Helper functions
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function showResult(message, type) {
    const div = document.getElementById("result");
    div.className = `result ${type}`;
    div.textContent = message;
    div.style.display = "block";
}
