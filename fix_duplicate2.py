with open('modules/fitur_anpr.py', 'rb') as f:
    content = f.read()

old = b'        }\n\n    def _bersihkan_plat(self, teks):\n        return "".join(ch for ch in str(teks).upper() if ch.isalnum'
new = b'        }\n'

if old in content:
    content = content.replace(old, new)
    with open('modules/fitur_anpr.py', 'wb') as f:
        f.write(content)
    print('Replaced successfully')
else:
    print('Pattern not found')