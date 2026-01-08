// Auto-typing
document.addEventListener('DOMContentLoaded', function(){
  const el = document.getElementById('typed');
  const text = "⏻Lucky AI — AI-Powered Medical Image Analysis — Research & Awareness demo for Brain MRI & Chest X-ray.";
  let idx = 0;
  function type() {
    if (!el) return;
    if (idx < text.length) {
      el.innerHTML += text[idx++];
      setTimeout(type, 18);
    }
  }
  type();
});

// file preview for uploader
document.addEventListener('change', function(e){
  if (e.target && e.target.id === 'fileInput') {
    const box = document.getElementById('previewBox');
    box.innerHTML = '';
    const file = e.target.files[0];
    if (!file) return;
    const img = document.createElement('img');
    img.style.maxWidth='320px';
    img.style.borderRadius='12px';
    img.style.boxShadow='0 20px 60px rgba(2,6,23,0.08)';
    img.src = URL.createObjectURL(file);
    box.appendChild(img);
  }
});
