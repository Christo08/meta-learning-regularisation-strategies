def format_duration(duration):
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    return f"{hours:02}:{minutes:02}:{seconds:02}"