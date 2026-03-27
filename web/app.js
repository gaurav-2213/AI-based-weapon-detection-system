const tabs = Array.from(document.querySelectorAll(".tab"));
const panelImage = document.getElementById("panel-image");
const panelVideo = document.getElementById("panel-video");

function setActive(tab) {
  for (const t of tabs) t.classList.toggle("active", t === tab);
  const which = tab.dataset.tab;
  panelImage.classList.toggle("hidden", which !== "image");
  panelVideo.classList.toggle("hidden", which !== "video");
}

tabs.forEach((t) => t.addEventListener("click", () => setActive(t)));

async function postFile(endpoint, file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(endpoint, { method: "POST", body: fd });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return await res.blob();
}

function setStatus(el, msg) {
  el.textContent = msg;
}

function renderImage(previewEl, blob) {
  previewEl.innerHTML = "";
  const url = URL.createObjectURL(blob);
  const img = document.createElement("img");
  img.src = url;
  img.alt = "Annotated output";
  previewEl.appendChild(img);
}

function renderVideo(previewEl, blob) {
  previewEl.innerHTML = "";
  const url = URL.createObjectURL(blob);
  const video = document.createElement("video");
  video.src = url;
  video.controls = true;
  video.playsInline = true;
  previewEl.appendChild(video);
}

const imageFile = document.getElementById("imageFile");
const runImage = document.getElementById("runImage");
const imageStatus = document.getElementById("imageStatus");
const imagePreview = document.getElementById("imagePreview");

runImage.addEventListener("click", async () => {
  const file = imageFile.files?.[0];
  if (!file) {
    setStatus(imageStatus, "Pick an image first.");
    return;
  }

  runImage.disabled = true;
  setStatus(imageStatus, "Running detection...");
  try {
    const out = await postFile("/detect-image", file);
    renderImage(imagePreview, out);
    setStatus(imageStatus, "Done. (Tip: right-click image to save.)");
  } catch (e) {
    setStatus(imageStatus, String(e?.message || e));
  } finally {
    runImage.disabled = false;
  }
});

const videoFile = document.getElementById("videoFile");
const runVideo = document.getElementById("runVideo");
const videoStatus = document.getElementById("videoStatus");
const videoPreview = document.getElementById("videoPreview");

runVideo.addEventListener("click", async () => {
  const file = videoFile.files?.[0];
  if (!file) {
    setStatus(videoStatus, "Pick a video first.");
    return;
  }

  runVideo.disabled = true;
  setStatus(videoStatus, "Processing video... (this can take a while)");
  try {
    const out = await postFile("/detect-video", file);
    renderVideo(videoPreview, out);
    setStatus(videoStatus, "Done. (Use the player menu to download if needed.)");
  } catch (e) {
    setStatus(videoStatus, String(e?.message || e));
  } finally {
    runVideo.disabled = false;
  }
});
