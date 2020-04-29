var spinner = document.getElementById("spinner")
spinner.style.display = "none";
var img = document.getElementById("img")
img.style.display = "none"

function displayImage(){
    spinner.style.display = "none";
    img.src = window.location.href + document.getElementById('fileName').textContent
    img.style.display = "block";
}

document.getElementById("inputGroupFileAddon02").onclick = async() => { 
    img.style.display = "none";
    spinner.style.display = "block";
    let image = document.getElementById("inputGroupFile02").files[0];
    let formData = new FormData();

    formData.append("file", image);    
    fetch(window.location.href, {method: "POST", body: formData}).then(() => {
        displayImage();
    }).catch(function(err) {
        console.log(err);
    });
} 

$(".custom-file-input").on("change", function() {
    var fileName = $(this).val().split("\\").pop();
    $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
  });

  