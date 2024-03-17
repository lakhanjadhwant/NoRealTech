// JavaScript to show MTECH profiles when MTECH link is clicked
function toggleStudentProfiles() {
    var profiles = document.getElementById("studentProfiles");
    var helloStudent = document.getElementById("helloStudent");
    if (profiles.style.display === "none" || profiles.style.display === "") {
      profiles.style.display = "block";
      helloStudent.style.display = "none"; // Hide Hello Student message
    } else {
      profiles.style.display = "none";
      helloStudent.style.display = "block"; // Show Hello Student message
    }
  }
  
  
  
  