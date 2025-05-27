from flask import Flask, render_template, request, redirect, url_for
import os
from image_info import generate_report_from_template, extract_metadata, calculate_sha256
from image_ela import perform_ela_analysis
from stegano import lsb
import platform
import webbrowser

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename.lower().endswith(('.jpg', '.jpeg')):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            # Run analysis
            meta = extract_metadata(filepath)
            meta_str = '\n'.join([f"{k}: {v}" for k, v in meta.items()]) if meta else "No metadata found."
            hash_val = calculate_sha256(filepath)
            perform_ela_analysis(filepath)
            ela_img = os.path.splitext(filepath)[0] + '_ELA.png'
            try:
                if not filepath.lower().endswith('.png'):
                    stego_result = "Steganography detection only works with PNG images."
                else:
                    hidden_message = lsb.reveal(filepath)
                    stego_result = f"Hidden message: {hidden_message}" if hidden_message else "No hidden message found."
            except Exception as e:
                    stego_result = f"Error: {e}"
            os_info = f"OS: {platform.system()} {platform.release()}"
            # Generate report
            generate_report_from_template(
                image_name=os.path.basename(filepath),
                hash_value=hash_val or "N/A",
                metadata=meta_str,
                ela_img=ela_img,
                stego_result=stego_result,
                os_info=os_info
            )
            return redirect(url_for('report', filename=os.path.basename(filepath)))
        else:
            return "Please upload a JPG image."
    return '''
    <h2>Upload an image for analysis</h2>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept=".jpg,.jpeg" required>
      <input type="submit" value="Analyze">
    </form>
    '''

@app.route('/report/<filename>')
def report(filename):
    with open("report.html", encoding="utf-8") as f:
        return f.read()

if __name__ == '__main__':
    app.run(debug=True)
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)