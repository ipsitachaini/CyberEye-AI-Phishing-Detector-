document.addEventListener("DOMContentLoaded", function () {
  // Initialize tooltips
  var tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });

  // Initialize progress bars animation
  animateProgressBars();

  // Setup file upload handling
  setupFileUpload();

  // Setup form validation
  setupFormValidation();

  // Auto-hide alerts after 5 seconds
  setTimeout(function () {
    var alerts = document.querySelectorAll(".alert-dismissible");
    alerts.forEach(function (alert) {
      var bsAlert = new bootstrap.Alert(alert);
      bsAlert.close();
    });
  }, 5000);
});

function animateProgressBars() {
  var progressBars = document.querySelectorAll(".progress-bar");
  progressBars.forEach(function (bar) {
    var width = bar.style.width;
    bar.style.width = "0%";
    setTimeout(function () {
      bar.style.transition = "width 1s ease-in-out";
      bar.style.width = width;
    }, 500);
  });
}

function setupFileUpload() {
  var fileInput = document.getElementById("file_input");
  if (fileInput) {
    fileInput.addEventListener("change", function (e) {
      var file = e.target.files[0];
      if (file) {
        var fileSize = file.size / (1024 * 1024); // Size in MB
        var maxSize = 50; // 50MB limit

        if (fileSize > maxSize) {
          showAlert("File size too large. Maximum size is 50MB.", "warning");
          fileInput.value = "";
          return;
        }

        var allowedTypes = [
          "text/plain",
          "text/csv",
          "application/json",
          "image/png",
          "image/jpeg",
          "image/gif",
          "image/bmp",
          "audio/mpeg",
          "audio/wav",
          "audio/ogg",
          "video/mp4",
          "video/avi",
          "video/quicktime",
        ];

        if (!allowedTypes.includes(file.type)) {
          showAlert(
            "File type not supported. Please check the allowed formats.",
            "warning"
          );
          fileInput.value = "";
          return;
        }

        // Show file info
        var fileInfo = document.createElement("div");
        fileInfo.className = "mt-2 text-muted small";
        fileInfo.innerHTML = `
                    <i class="fas fa-file"></i> ${
                      file.name
                    } (${fileSize.toFixed(2)} MB)
                `;

        // Remove any existing file info
        var existing = fileInput.parentNode.querySelector(".file-info");
        if (existing) existing.remove();

        fileInfo.className += " file-info";
        fileInput.parentNode.appendChild(fileInfo);
      }
    });
  }
}

function setupFormValidation() {
  var forms = document.querySelectorAll(".needs-validation");
  forms.forEach(function (form) {
    form.addEventListener("submit", function (e) {
      if (!form.checkValidity()) {
        e.preventDefault();
        e.stopPropagation();
      }
      form.classList.add("was-validated");
    });
  });

  // Text area character counter
  var textInput = document.getElementById("text_input");
  if (textInput) {
    var counter = document.createElement("div");
    counter.className = "text-muted small mt-1";
    counter.id = "char-counter";
    textInput.parentNode.appendChild(counter);

    textInput.addEventListener("input", function () {
      var length = textInput.value.length;
      var maxLength = 5000;
      counter.innerHTML = `${length}/${maxLength} characters`;

      if (length > maxLength * 0.9) {
        counter.className = "text-warning small mt-1";
      } else {
        counter.className = "text-muted small mt-1";
      }

      if (length >= maxLength) {
        counter.className = "text-danger small mt-1";
        counter.innerHTML += " (Maximum reached)";
      }
    });
  }
}

function showAlert(message, type) {
  var alertContainer = document.querySelector(".container");
  var alert = document.createElement("div");
  alert.className = `alert alert-${type} alert-dismissible fade show`;
  alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

  alertContainer.insertBefore(alert, alertContainer.firstChild);

  // Auto-hide after 5 seconds
  setTimeout(function () {
    var bsAlert = new bootstrap.Alert(alert);
    bsAlert.close();
  }, 5000);
}

function showLoading(element) {
  var spinner = document.createElement("div");
  spinner.className = "spinner";
  element.disabled = true;
  element.innerHTML = '<div class="spinner"></div> Processing...';
}

function hideLoading(element, originalText) {
  element.disabled = false;
  element.innerHTML = originalText;
}

// API Helper functions
async function analyzeTextAPI(text) {
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: text }),
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
}

async function getHistoryAPI() {
  try {
    const response = await fetch("/api/history");
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
}

// Utility functions
function formatDate(dateString) {
  const date = new Date(dateString);
  return date.toLocaleDateString() + " " + date.toLocaleTimeString();
}

function formatFileSize(bytes) {
  const sizes = ["Bytes", "KB", "MB", "GB"];
  if (bytes === 0) return "0 Bytes";
  const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
  return Math.round(bytes / Math.pow(1024, i), 2) + " " + sizes[i];
}

function getSafetyColor(percentage) {
  if (percentage >= 80) return "success";
  if (percentage >= 60) return "warning";
  return "danger";
}

function getLabelColor(label) {
  switch (label.toLowerCase()) {
    case "safe":
      return "success";
    case "suspicious":
      return "warning";
    case "phishing":
      return "danger";
    default:
      return "secondary";
  }
}

// Dark mode toggle (future feature)
function toggleDarkMode() {
  document.body.classList.toggle("dark-mode");
  localStorage.setItem(
    "darkMode",
    document.body.classList.contains("dark-mode")
  );
}

// Export functionality
function exportResults(format) {
  // This would export detection results in various formats
  showAlert("Export feature will be available in future updates.", "info");
}

// Real-time updates (using WebSocket in production)
function initializeRealTimeUpdates() {
  // This would set up WebSocket connections for real-time updates
  console.log("Real-time updates initialized");
}
