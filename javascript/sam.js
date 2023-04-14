function getRealCoordinate(image, x1, y1) {
    if (image.naturalHeight * (image.width / image.naturalWidth) <= image.height) {
        // width is filled, height has padding
        const scale = image.naturalWidth / image.width
        const zero_point = (image.height - image.naturalHeight / scale) / 2
        const x = x1 * scale
        const y = (y1 - zero_point) * scale
        return [x, y]
    } else {
        // height is filled, width has padding
        const scale = image.naturalHeight / image.height
        const zero_point = (image.width - image.naturalWidth / scale) / 2
        const x = (x1 - zero_point) * scale
        const y = y1 * scale
        return [x, y]
    }
}

function changeRunButton() {
    const sam_run_button = gradioApp().getElementById("sam_run_button");
    const sam_mode = (samHasImageInput() && (samCanSubmit() || (dinoCanSubmit() && dinoPreviewCanSubmit()))) ? "block" : "none";
    if (sam_run_button && sam_run_button.style.display != sam_mode) {
        sam_run_button.style.display = sam_mode;
    }

    const dino_run_button = gradioApp().getElementById("dino_run_button");
    const dino_mode = (samHasImageInput() && dinoCanSubmit()) ? "block" : "none";
    if (dino_run_button && dino_run_button.style.display != dino_mode) {
        dino_run_button.style.display = dino_mode;
    }
}

function registerDinoTextObserver() {
    const dino_text_prompt = gradioApp().getElementById("dino_text_prompt").querySelector("textarea")
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.target === dino_text_prompt) {
                changeRunButton();
            }
        });
    });
    observer.observe(dino_text_prompt, { attributes: true });
    return arguments;
}

function switchToInpaintUpload() {
    switch_to_img2img_tab(4)
    return arguments;
}

function immediatelyGenerate() {
    const runButton = gradioApp().getElementById("sam_run_button");
    if (runButton.style.display !== "none") {
        runButton.click();

    }
}
function isRealTimePreview() {
    return gradioApp().querySelector("#sam_realtime_preview_checkbox input[type='checkbox']").checked;
}

function createDot(sam_image, image, coord, label) {
    const x = coord.x;
    const y = coord.y;
    const realCoord = getRealCoordinate(image, coord.x, coord.y);
    if (realCoord[0] >= 0 && realCoord[0] <= image.naturalWidth && realCoord[1] >= 0 && realCoord[1] <= image.naturalHeight) {
        const circle = document.createElement("div");
        circle.style.position = "absolute";
        circle.style.width = "10px";
        circle.style.height = "10px";
        circle.style.borderRadius = "50%";
        circle.style.backgroundColor = label == "sam_positive" ? "black" : "red";
        circle.style.left = x + "px";
        circle.style.top = y + "px";
        circle.className = label;
        circle.title = (label == "sam_positive" ? "positive" : "negative") + "point label, left click it to cancel.";
        sam_image.appendChild(circle);
        circle.addEventListener("click", e => {
            e.stopPropagation();
            circle.remove();
            if (gradioApp().querySelectorAll(".sam_positive").length == 0 &&
                gradioApp().querySelectorAll(".sam_negative").length == 0) {
                disableRunButton();
            } else {
                if (isRealTimePreview()) {
                    immediatelyGenerate();
                }
            }
        });
        enableRunButton();
        if (isRealTimePreview()) {
            immediatelyGenerate();
        }
    }
}

function removeDots() {
    const sam_image = gradioApp().getElementById("sam_input_image");
    if (sam_image) {
        [".sam_positive", ".sam_negative"].forEach(cls => {
            const dots = sam_image.querySelectorAll(cls);
    
            dots.forEach(dot => {
                dot.remove();
            });
        })
    }
    return arguments;
}

function create_submit_sam_args(args) {
    res = []
    for (var i = 0; i < args.length; i++) {
        res.push(args[i])
    }

    res[res.length - 1] = null

    return res
}


function submit_dino() {
    res = []
    for (var i = 0; i < arguments.length; i++) {
        res.push(arguments[i])
    }

    res[res.length - 2] = null
    res[res.length - 1] = null
    return res
}

function submit_sam() {
    let res = create_submit_sam_args(arguments);
    let positive_points = [];
    let negative_points = [];
    const sam_image = gradioApp().getElementById("sam_input_image");
    const image = sam_image.querySelector('img');
    const classes = [".sam_positive", ".sam_negative"];
    classes.forEach(cls => {
        const dots = sam_image.querySelectorAll(cls);
        dots.forEach(dot => {
            const width = parseFloat(dot.style["left"]);
            const height = parseFloat(dot.style["top"]);
            if (cls == ".sam_positive") {
                positive_points.push(getRealCoordinate(image, width, height));
            } else {
                negative_points.push(getRealCoordinate(image, width, height));
            }
        });
    });
    res[2] = positive_points;
    res[3] = negative_points;
    return res
}

function dinoPreviewCanSubmit() {
    const dino_preview_enable_checkbox = gradioApp().getElementById("dino_preview_checkbox");
    if (!dino_preview_enable_checkbox || 
        (dino_preview_enable_checkbox.querySelector("input") &&
        !dino_preview_enable_checkbox.querySelector("input").checked)) {
        return true;
    } else {
        let dino_preview_selected = false;
        gradioApp().getElementById("dino_preview_boxes_selection").querySelectorAll("input").forEach(element => dino_preview_selected = element.checked || dino_preview_selected);
        return dino_preview_selected;
    }
}

function dinoCanSubmit() {
    const dino_enable_checkbox = gradioApp().getElementById("dino_enable_checkbox")
    const dino_text_prompt = gradioApp().getElementById("dino_text_prompt")
    return (dino_enable_checkbox && dino_text_prompt &&
        dino_enable_checkbox.querySelector("input") &&
        dino_enable_checkbox.querySelector("input").checked &&
        dino_text_prompt.querySelector("textarea") &&
        dino_text_prompt.querySelector("textarea").value != "")
}

function samCanSubmit() {
    return (gradioApp().querySelectorAll(".sam_positive").length > 0 ||
        gradioApp().querySelectorAll(".sam_negative").length > 0)
}

function samHasImageInput() {
    const sam_image = gradioApp().getElementById("sam_input_image")
    return sam_image && sam_image.querySelector('img')
}

function onChangeDinoPreviewBoxesSelection() {
    changeRunButton(arguments[0].length > 0)
}

prevImg = null

onUiUpdate(() => {
    const sam_image = gradioApp().getElementById("sam_input_image")
    if (sam_image) {
        const image = sam_image.querySelector('img')
        if (image && prevImg != image.src) {
            removeDots();
            prevImg = image.src;

            image.addEventListener("click", event => {
                const rect = event.target.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;

                createDot(sam_image, event.target, { x, y }, "sam_positive");
            });

            image.addEventListener("contextmenu", event => {
                event.preventDefault();
                const rect = event.target.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;

                createDot(sam_image, event.target, { x, y }, "sam_negative");
            });

            const observer = new MutationObserver(mutations => {
                mutations.forEach(mutation => {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'src' && mutation.target === image) {
                        removeDots();
                        prevImg = image.src;
                    }
                });
            });

            observer.observe(image, { attributes: true });
        } else if (!image) {
            removeDots();
            prevImg = null;
        }
    }

    changeRunButton();
})
