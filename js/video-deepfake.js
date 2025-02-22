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

