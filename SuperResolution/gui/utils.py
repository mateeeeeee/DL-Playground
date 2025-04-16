from PIL import Image, ImageTk

def display_image_in_label(image_pil, label_widget, max_size):
    if image_pil is None:
        label_widget.config(image='')
        label_widget.image = None
        return None

    def _calculate_thumbnail_size(original_size, max_size):
        ow, oh = original_size
        if ow == 0 or oh == 0: return (0,0) 
        mw, mh = max_size
        ratio = min(mw / ow, mh / oh) if ow > 0 and oh > 0 else 1.0
        return max(1, int(ow * ratio)), max(1, int(oh * ratio))

    thumb_w, thumb_h = _calculate_thumbnail_size(image_pil.size, max_size)

    if thumb_w <= 0 or thumb_h <= 0:
        label_widget.config(image='')
        label_widget.image = None
        return None

    try:
        display_img_thumb = image_pil.copy()
        display_img_thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(display_img_thumb)
        label_widget.config(image=img_tk)
        label_widget.image = img_tk 
        return display_img_thumb 
    except Exception as e:
        print(f"Error creating thumbnail or displaying image: {e}")
        label_widget.config(image='')
        label_widget.image = None
        return None
