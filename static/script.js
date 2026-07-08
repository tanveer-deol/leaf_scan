const input = document.getElementById("image");
const preview = document.getElementById("preview");

input.onchange = () => {
    if (!input.files || !input.files[0]) {
        alert("Please select an image first.");
        return;
    }

    preview.src = URL.createObjectURL(input.files[0]);
};

async function predict() {

    let file = input.files[0];

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    const btn = document.getElementById("predictBtn");
    const loading = document.getElementById("loading");

    // Show loading state
    btn.disabled = true;
    btn.innerHTML = "Analyzing...";
    loading.style.display = "flex";

    try {

        let form = new FormData();
        form.append("image", file);

        let response = await fetch("/predict", {
            method: "POST",
            body: form
        });

        let result = await response.json();
        document.getElementById("result").innerHTML = `
    <h3>${result.prediction}</h3>
    <p><strong>Confidence:</strong> ${result.confidence}%</p>

    <hr>

    <p><strong>Cause:</strong><br>${result.cause}</p>

    <p><strong>Prevention:</strong><br>${result.prevention}</p>

    <p><strong>Treatment:</strong><br>${result.treatment}</p>
`;


    } catch (error) {

        alert("Prediction failed. Please try again.");
        console.error(error);

    } finally {

        // Restore button
        btn.disabled = false;
        btn.innerHTML = "Predict Uploaded Image";
        loading.style.display = "none";

    }
}
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

let stream = null;

// Open camera only when button is clicked
document.getElementById("openCamera").onclick = async () => {

    stream = await navigator.mediaDevices.getUserMedia({
    video: {
        facingMode: {
            ideal: "environment"
        }
    },
    audio: false
});

    video.srcObject = stream;

    video.style.display = "block";
    document.getElementById("capture").style.display = "inline-block";
};

// Capture image
document.getElementById("capture").onclick = () => {

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {

        const formData = new FormData();
        formData.append("image", blob, "capture.png");

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        if (result.prediction.startsWith("Uncertain")) {
    document.getElementById("result").innerHTML = `
        <span style="color:#e67e22;font-weight:bold;">
            ${result.prediction}
        </span><br>
        Confidence: ${result.confidence}%
        `;
        } else {
            document.getElementById("result").innerHTML = `
                <span style="color:#2e7d32;font-weight:bold;">
                    ${result.prediction}
                </span><br>
                Confidence: ${result.confidence}%
        `;
        }

        // Stop camera after capture
        stream.getTracks().forEach(track => track.stop());

        video.style.display = "none";
        document.getElementById("capture").style.display = "none";

    }, "image/png");
};