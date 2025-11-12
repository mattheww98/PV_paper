from codecarbon import OfflineEmissionsTracker
import subprocess
out_dir = '.'
tracker = OfflineEmissionsTracker(
            output_dir=out_dir,
            country_iso_code="GBR"
        )

tracker.start()
subprocess.run("./VASP_run.sh", shell=True)

tracker.stop()
