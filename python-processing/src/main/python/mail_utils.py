def mail_preprocessor(doc):
    """Return only the Subject and Content parts from the mail."""
    accept = False
    accepted_lines = []
    for line in doc.splitlines():
        l = line.lower()
        if l.startswith('subject'):
            accept = True
            line = line.lstrip('subject')
        elif l.startswith('content'):
            accept = True
            line = line.lstrip('content')
        if accept:
            accepted_lines.append(line.strip())
    return '\n'.join(accepted_lines)