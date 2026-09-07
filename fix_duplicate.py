with open('modules/fitur_anpr.py', 'rb') as f:
    content = f.read()

old = b'ending": True,\n        }\n\n    def _bersihkan_plat(self, teks):\n        return "".join(ch for ch in str(teks).upper() if ch.isalnum())\n\n    def bersihkan_memori'
new = b'ending": True,\n        }\n\n    def bersihkan_memori'

if old in content:
    content = content.replace(old, new)
    with open('modules/fitur_anpr.py', 'wb') as f:
        f.write(content)
    print('Replaced successfully')
else:
    print('Pattern not found')