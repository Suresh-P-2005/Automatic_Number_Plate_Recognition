const webcam = document.getElementById(
    "webcam"
);

const processedFrame = document.getElementById(
    "processed-frame"
);

const startButton = document.getElementById(
    "start-camera"
);

const stopButton = document.getElementById(
    "stop-camera"
);

const canvas = document.getElementById(
    "capture-canvas"
);

const statusText = document.getElementById(
    "detection-status"
);

const resultTable = document.getElementById(
    "result-table"
);

const clearResultsButton =
    document.getElementById(
        "clear-results"
    );



let cameraStream = null;

let processingTimer = null;

let isProcessing = false;

let savedDetections = {};

async function startWebcam() {

    try {

        cameraStream =
            await navigator.mediaDevices
            .getUserMedia(
                {
                    video: {
                        width: {
                            ideal: 1280
                        },

                        height: {
                            ideal: 720
                        }
                    },

                    audio: false
                }
            );


        webcam.srcObject =
            cameraStream;


        startButton.disabled =
            true;


        stopButton.disabled =
            false;


        statusText.textContent =
            "Webcam started. Waiting for vehicle...";


        processingTimer =
            setInterval(
                sendFrameToANPR,
                500
            );

    }

    catch (error) {

        console.error(
            error
        );


        statusText.textContent =
            "Could not access webcam. "
            + "Allow camera permission "
            + "and try again.";

    }

}


function stopWebcam() {

    if (
        processingTimer
    ) {

        clearInterval(
            processingTimer
        );

        processingTimer = null;

    }


    if (
        cameraStream
    ) {

        cameraStream
            .getTracks()
            .forEach(
                track => {

                    track.stop();

                }
            );

        cameraStream = null;

    }


    webcam.srcObject =
        null;


    processedFrame.removeAttribute(
        "src"
    );


    startButton.disabled =
        false;


    stopButton.disabled =
        true;


    statusText.textContent =
        "Webcam stopped.";


    isProcessing =
        false;

}


async function sendFrameToANPR() {

    if (
        !cameraStream
        || isProcessing
        || webcam.videoWidth === 0
    ) {

        return;

    }


    isProcessing =
        true;


    canvas.width =
        webcam.videoWidth;


    canvas.height =
        webcam.videoHeight;


    const context =
        canvas.getContext(
            "2d"
        );


    context.drawImage(
        webcam,
        0,
        0,
        canvas.width,
        canvas.height
    );


    canvas.toBlob(
        async function (
            blob
        ) {

            if (
                !blob
            ) {

                isProcessing =
                    false;

                return;

            }


            const formData =
                new FormData();


            formData.append(
                "frame",
                blob,
                "webcam_frame.jpg"
            );


            try {

                const response =
                    await fetch(
                        "/process-webcam-frame",
                        {
                            method:
                                "POST",

                            body:
                                formData
                        }
                    );


                if (
                    !response.ok
                ) {

                    throw new Error(
                        "ANPR processing failed"
                    );

                }


                const data =
                    await response.json();


                processedFrame.src =
                    "data:image/jpeg;base64,"
                    + data.image;


                updateResults(
                    data.results
                );


                statusText.textContent =
                    data.status;

            }

            catch (
                error
            ) {

                console.error(
                    error
                );


                statusText.textContent =
                    "Error processing webcam frame.";

            }

            finally {

                isProcessing =
                    false;

            }

        },

        "image/jpeg",

        0.85
    );

}


function updateResults(
    results
) {

    /*
    Store only valid plate results.
    Do not remove old results when
    one webcam frame fails.
    */

    if (
        results
        && results.length > 0
    ) {

        results.forEach(
            result => {

                const plate =
                    result.license_plate;


                /*
                Ignore temporary
                OCR results.
                */

                if (
                    plate
                    && plate !== "Reading..."
                    && plate !== "Unknown"
                    && plate !== "Not detected"
                ) {

                    savedDetections[
                        plate
                    ] = {

                        vehicle_id:
                            result.vehicle_id,

                        license_plate:
                            plate,

                        number_of_readings:
                            result.number_of_readings

                    };

                }

            }
        );

    }


    /*
    Clear table only before
    displaying saved results.
    */

    resultTable.innerHTML =
        "";


    const savedResults =
        Object.values(
            savedDetections
        );


    /*
    If no valid plate has ever
    been detected.
    */

    if (
        savedResults.length === 0
    ) {

        resultTable.innerHTML =
            `
            <tr>

                <td colspan="3">

                    Waiting for a valid
                    licence plate...

                </td>

            </tr>
            `;

        return;

    }


    /*
    Display all saved plates.
    */

    savedResults.forEach(
        result => {

            const row =
                document.createElement(
                    "tr"
                );


            row.innerHTML =
                `
                <td>

                    ${result.vehicle_id}

                </td>

                <td>

                    ${result.license_plate}

                </td>

                <td>

                    ${result.number_of_readings}

                </td>
                `;


            resultTable.appendChild(
                row
            );

        }
    );

}


startButton.addEventListener(
    "click",
    startWebcam
);


stopButton.addEventListener(
    "click",
    stopWebcam
);

clearResultsButton.addEventListener(
    "click",
    function () {

        savedDetections = {};


        resultTable.innerHTML =
            `
            <tr>

                <td colspan="3">

                    Results cleared.

                </td>

            </tr>
            `;

    }
);

window.addEventListener(
    "beforeunload",
    stopWebcam
);