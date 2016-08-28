"""
bla
"""
def generate_html(filename=None):
    #convert source code files into formatted html for documentation
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
            print ("html_filename = " , html_filename)
            fid.write(header_str)
            fid.write("<code>")
            in_comment_flag = 0
            for line in fid_source:
                if line.lstrip().startswith("#"): # then this line is a single line comment and we don't format it
                    fid.write(line.replace('\n',"<br>"))
                elif line.lstrip().startswith("\"\"\""): # ignore formatting on multi-line comments
                    fid.write(line.replace('\n',"<br>"))
                    in_comment_flag = (in_comment_flag + 1) % 2
                elif in_comment_flag:
                    fid.write(line.replace('\n',"<br>"))
                else:
                    inline_comment_start = line.find("#")
                    if inline_comment_start == -1: # no inline comment
                        fid.write(code_formatter(line))
                    else:
                        fid.write(code_formatter(line[:inline_comment_start].replace('\n',"<br>")))
                        fid.write(line[inline_comment_start:].replace('\n',"<br>"))
            fid.write("</code>")
            fid.write(closing_str)

def code_formatter(string):
    # A simple custom HTML code formatter that colors some keywords and tries to avoid instances of them within comments
    color_string = "\"orange\""
    keywords = ["from", "def","class", "self", "__main__", "for", "if",\
                "while","import","else","elif", "True","False", "del", \
                "with", "plt", "QtCore", "QtGui", "Qt", "ui"]
    short_keywords = keywords[:]
    new_keywords = []
    for i, v in enumerate(keywords):
        keywords[i] = v + " "
        new_keywords.append(v + ".")
        new_keywords.append(v + "\n")
    keywords += new_keywords
    format_string = ("<font color=" + color_string + ">", "</font>")
    replacers={}

    for kw in keywords:
        replacers[kw] = format_string

    GENFIRE_format_string = ("<b><font color=" + color_string + ">", "</font></b>")
    replacers['GENFIRE'] = GENFIRE_format_string
    replacers['genfire'] = GENFIRE_format_string



    # replacers={"from":("<font color=" + color_string + ">", "</font>")}



    string = string.replace('\t', "&nbsp&nbsp&nbsp&nbsp")
    if replacers is not None:
        for key, value in replacers.iteritems():
            # print (key, value)
            string = string.replace(key,value[0] + key + value[1])

    # for key in short_keywords:
    #     if string.endswith(key):
    #         string.replace(key, ("<font color=" + color_string + ">" + key+ "</font>") )
    string = string.replace('\n',"<br>")
    return string

if __name__=="__main__":
    import sys
    if len(sys.argv) > 1:
        for j in range(1,len(sys.argv)):
            generate_html(sys.argv[j])
    else:
        generate_html("html_helper.py")
        