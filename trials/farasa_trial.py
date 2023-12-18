import subprocess

# Path to the Farasa POS JAR file
farasa_jar_path = "/path/to/FarasaPOS.jar"

# Example Arabic text
arabic_text = "اللغة العربية جميلة وغنية بالتنوع."

# Run Farasa POS tagging using subprocess
process = subprocess.Popen(['java', '-jar', farasa_jar_path, '-t', arabic_text], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output, error = process.communicate()

# Display the result
print(output.decode('utf-8'))
