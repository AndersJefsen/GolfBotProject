import win32gui

def enum_windows_callback(hwnd, windows_list):
    # Append the window handle and window text (title) to the list
    if win32gui.IsWindowVisible(hwnd):
        window_text = win32gui.GetWindowText(hwnd)
        if window_text:  # Only consider windows with a non-empty title
            windows_list.append((hwnd, window_text))

def list_open_windows():
    windows_list = []
    # Enumerate all top-level windows and call the callback function for each
    win32gui.EnumWindows(enum_windows_callback, windows_list)
    return windows_list

if __name__ == "__main__":
    open_windows = list_open_windows()
    for hwnd, window_text in open_windows:
        print(f"Handle: {hwnd}, Title: {window_text}")