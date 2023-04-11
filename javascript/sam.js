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

function enableRunButton() {
    gradioApp().getElementById("sam_run_button").style.display = "block";
}

function disableRunButton() {
    gradioApp().getElementById("sam_run_button").style.display = "none";
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
            }
        });
        enableRunButton();
    }
}

function removeDots(parentDiv) {
    [".sam_positive", ".sam_negative"].forEach(cls => {
        const dots = parentDiv.querySelectorAll(cls);
    
        dots.forEach(dot => {
            dot.remove();
        });
    })
    disableRunButton();
}

function create_submit_sam_args(args) {
    res = []
    for (var i = 0; i < args.length; i++) {
        res.push(args[i])
    }

    res[res.length - 1] = null
    res[res.length - 2] = null

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
    res[3] = positive_points;
    res[4] = negative_points;
    return res
}

prevImg = null

onUiUpdate(() => {
    const sam_image = gradioApp().getElementById("sam_input_image")
    if (sam_image) {
        const image = sam_image.querySelector('img')
        if (image && prevImg != image.src) {
            removeDots(sam_image);
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
                        removeDots(sam_image);
                        prevImg = image.src;
                    }
                });
            });

            observer.observe(image, { attributes: true });
        } else if (!image) {
            removeDots(sam_image);
            prevImg = null;
        }
    }
})