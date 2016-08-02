def generate_html(filename=None):
    import sys
    import os
    print(filename)
    if not os.path.isfile(filename):
        raise NameError("File Not Found")

    # if filename is None and len(sys.argv)>1:
    #     filename=sys.argv[1]

    html_filename = os.path.abspath(os.path.splitext(filename)[0]) + '.html'

    header_str = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>""" + os.path.splitext(filename)[0] + """</title>
  <meta name="description" content="GENFIRE Documentation">
  <meta name="author" content="Alan (AJ) Pryor, Jr.">\
  <link rel="stylesheet" href="css/styles.css?v=1.0">
  <link rel="stylesheet" type="text/css" href="main.css">
</head>
<body>\n"""

    closing_str = """</body>
    </html>"""
    print("hs = {}".format(header_str))

    with open(filename,'r') as fid_source:
        with open(html_filename,'w') as fid:
            fid.write(header_str)
            fid.write("<code>")
            fid.write(code_formatter(fid_source.read()))
            fid.write("</code>")
            fid.write(closing_str)

def code_formatter(input):
    color_string = "\"green\""
    keywords = ["from", "def","class", "self", "__main__", "for", "if", "while","import"]
    format_string = ("<font color=" + color_string + ">", "</font>")
    replacers={}
    for kw in keywords:
        replacers[kw] = format_string
    # replacers={"from":("<font color=" + color_string + ">", "</font>")}


    string = input.replace('\n',"<br>")
    string = string.replace('\t', "&nbsp&nbsp&nbsp&nbsp")
    if replacers is not None:
        for key, value in replacers.iteritems():
            print (key, value)
            string = string.replace(key,value[0] + key + value[1])
    return string

if __name__=="__main__":
    import sys
    if len(sys.argv) > 1:
        for j in range(1,len(sys.argv)):
            generate_html(sys.argv[j])
    else:
        generate_html("test.py")
        