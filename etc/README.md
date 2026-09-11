# Configuration

`spec.dat` is an editable example of the JUST instrument configuration.
The identical installed default is `py/just_etc/data/spec.dat`; package data
is used because pip does not automatically install top-level `etc/` files.
Update both copies together when changing the instrument model.

Pass a custom file with `JUSTExposureTimeCalculator(spec_file="/path/to/spec.dat")`.
