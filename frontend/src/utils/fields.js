/**
 * Feature definitions for the PCOS textual prediction form.
 *
 * These 13 fields are the complete feature set the model is trained on
 * (plus BMI, which the backend derives from weight and height).
 *
 * Lab values (FSH, LH, beta-HCG, follicle counts) are deliberately NOT here:
 * the model no longer uses them. It previously trained on all 41 spreadsheet
 * columns and median-filled the missing 28 at inference, which dropped
 * sensitivity to 0.26. See backend/app/services/textual_service.py.
 *
 * Field names MUST match the dataset column names exactly
 * (the backend strips whitespace, so minor spacing is OK).
 */

export const FIELDS = [
    // Demographics
    { name: "Age (yrs)", label: "Age (years)", type: "number", group: "Demographics", min: 10, max: 60 },
    { name: "Weight (Kg)", label: "Weight (kg)", type: "number", group: "Demographics", step: 0.1, min: 25, max: 200 },
    { name: "Height(Cm)", label: "Height (cm)", type: "number", group: "Demographics", step: 0.1, min: 100, max: 220 },
    {
        name: "Blood Group",
        label: "Blood Group",
        type: "select",
        group: "Demographics",
        options: [
            { value: "11", label: "A+" },
            { value: "12", label: "A−" },
            { value: "13", label: "B+" },
            { value: "14", label: "B−" },
            { value: "15", label: "O+" },
            { value: "16", label: "O−" },
            { value: "17", label: "AB+" },
            { value: "18", label: "AB−" },
        ],
    },

    // Clinical / Menstrual
    {
        name: "Cycle(R/I)",
        label: "Cycle Regularity",
        type: "select",
        group: "Menstrual",
        options: [
            { value: "2", label: "Regular" },
            { value: "4", label: "Irregular" },
        ],
    },
    { name: "Cycle length(days)", label: "Cycle Length (days)", type: "number", group: "Menstrual", min: 14, max: 180, defaultValue: 28 },

    // Symptoms (all Y/N)
    { name: "Weight gain(Y/N)", label: "Weight Gain", type: "yesno", group: "Symptoms" },
    { name: "hair growth(Y/N)", label: "Excess Hair Growth", type: "yesno", group: "Symptoms" },
    { name: "Skin darkening (Y/N)", label: "Skin Darkening", type: "yesno", group: "Symptoms" },
    { name: "Hair loss(Y/N)", label: "Hair Loss", type: "yesno", group: "Symptoms" },
    { name: "Pimples(Y/N)", label: "Pimples / Acne", type: "yesno", group: "Symptoms" },

    // Lifestyle
    { name: "Fast food (Y/N)", label: "Fast Food Consumption", type: "yesno", group: "Lifestyle" },
    { name: "Reg.Exercise(Y/N)", label: "Regular Exercise", type: "yesno", group: "Lifestyle" },
];

export const GROUPS = [
    { key: "Demographics", label: "Demographics" },
    { key: "Menstrual", label: "Menstrual Health" },
    { key: "Symptoms", label: "Symptoms" },
    { key: "Lifestyle", label: "Lifestyle" },
];

/**
 * Returns a default values object for all fields.
 */
export function getDefaultValues() {
    const defaults = {};
    FIELDS.forEach((f) => {
        if (f.defaultValue !== undefined) {
            defaults[f.name] = String(f.defaultValue);
        } else if (f.type === "yesno") {
            defaults[f.name] = "0";
        } else if (f.type === "select" && f.options?.length) {
            defaults[f.name] = f.options[0].value;
        } else {
            defaults[f.name] = "";
        }
    });
    return defaults;
}

/**
 * Build the `features` payload for the prediction endpoints.
 *
 * Values that are not valid numbers are omitted rather than coerced to 0 —
 * the backend rejects missing fields explicitly, so a blank Age surfaces as an
 * error instead of silently being sent as a real measurement of zero.
 */
export function toFeaturePayload(values) {
    const features = {};
    Object.entries(values).forEach(([key, val]) => {
        const num = parseFloat(val);
        if (Number.isFinite(num)) features[key] = num;
    });
    return features;
}