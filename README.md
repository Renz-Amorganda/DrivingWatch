# DrivingWatch: Operator Fatigue and Drowsiness Detector

An on-device computer vision web application engineered for real-time operator fatigue and drowsiness detection. Built using Google Teachable Machine and TensorFlow.js, this system analyzes video feeds entirely within the browser via WebGL shaders, ensuring zero latency and total data privacy.

---

## 1. Project Overview

DrivingWatch addresses the danger of operator fatigue during driving and prolonged workstation sessions. The system classifies video input into three states:

* **Alert and Focused:** Head upright, eyes open, attentive posture.
* **Drowsy / Nodding:** Eyes closed, head nodding downward, or chin drooping.
* **Distracted:** Head turned away from the primary field of view.

### Key Capabilities

* **Client-Side Neural Processing:** Video frames never leave the client machine. Zero server bandwidth or cloud API latency.
* **Fatigue Hysteresis Engine:** Integrates a continuous 1.5-second time threshold buffer to filter out natural eye blinks and prevent false alarms.
* **Acoustic Warning System:** Synthesizes low-latency warning tones via the Web Audio API without requiring external audio assets.
* **Telemetry Dashboard:** Live FPS counter, inference latency in milliseconds, session duration, and real-time incident event logging.
* **GitHub Pages Ready:** Pure static architecture (HTML, CSS, JavaScript) deployable without build steps.

---

## 2. Step-by-Step: Training Your Model in Google Teachable Machine

You can train your custom model in less than five minutes:

1. Visit [Google Teachable Machine](https://teachablemachine.withgoogle.com/) and select **Get Started**, then choose **Image Project** > **Standard Image Model**.
2. Define three classes:
   * **Class 1: `Alert`**
     * Click **Webcam**, hold down **Record**, and capture 150 to 200 samples of yourself sitting upright, looking straight ahead, blinking normally, and making minor natural head movements.
   * **Class 2: `Drowsy`**
     * Click **Webcam** and record 150 to 200 samples with your eyes closed, chin dipping downward, or head slumped forward as if falling asleep.
   * **Class 3: `Distracted`**
     * Record 150 to 200 samples looking far to the left, far to the right, or downward toward a desk surface.
3. Click **Train Model** (keep the browser tab open while training completes).
4. Click **Export Model**:
   * Select **TensorFlow.js**.
   * Under the **Upload (shareable link)** tab, click **Upload my model**.
   * Copy the generated link (e.g., `https://teachablemachine.withgoogle.com/models/xxxxxxxxx/`).
5. In the DrivingWatch web app, click **Model Source** in the top navigation, paste your URL, and click **Load Custom Model**.

---

## 3. Local Verification

To run the application locally without external build tools:

Using Python:
```bash
python -m http.server 8000
```

Using Node / npx:
```bash
npx serve .
```

Open `http://localhost:8000` in any modern web browser (Chrome, Edge, Safari, Firefox).

---

## 4. Deploying to GitHub Pages

1. Initialize a git repository and commit your files:
   ```bash
   git init
   git add .
   git commit -m "Initial release of DrivingWatch drowsiness detector"
   ```
2. Create a repository on GitHub named `DrivingWatch` or `drivingwatch`.
3. Push your repository:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```
4. On GitHub, navigate to **Settings** > **Pages**.
5. Under **Build and deployment**, select **Deploy from a branch**.
6. Set the branch to `main` and folder to `/ (root)`. Click **Save**.
7. Your application will be live within one minute at `https://YOUR_USERNAME.github.io/REPO_NAME/`.

---

## 5. System Compliance and Architecture

* **Privacy:** Video frames are processed in-memory and discarded immediately after shader evaluation.
* **Performance:** Targets 60 FPS on standard modern integrated GPUs via WebGL hardware acceleration.
* **Accessibility:** Full keyboard dismissal for modal sheets, high-contrast indicators, and configurable auditory alerts.
