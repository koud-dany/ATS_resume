document.getElementById("pdf-upload").addEventListener('change', function() {
    var file = this.files[0];
    var reader = new FileReader();
    reader.onload = function() {
      // Use pdf.js library to extract text content from PDF file
      pdfjsLib.getDocument({data: new Uint8Array(this.result)}).promise.then(function(pdf) {
        pdf.getPage(1).then(function(page) {
          page.getTextContent().then(function(textContent) {
            var text = '';
            for (var i = 0; i < textContent.items.length; i++) {
              text += textContent.items[i].str + ' ';
            }
            document.getElementById('text1').value = text;
          });
        });
      });
    };
    reader.readAsArrayBuffer(file);
  });