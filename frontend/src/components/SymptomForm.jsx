import { FIELDS, GROUPS } from "../utils/fields";

/**
 * Reusable symptom form component.
 *
 * Props:
 *  - values: object — controlled form state keyed by field name
 *  - onChange: (name, value) => void
 */
export default function SymptomForm({ values, onChange }) {
  const handleChange = (e) => {
    onChange(e.target.name, e.target.value);
  };

  return (
    <div className="symptom-form">
      {GROUPS.map((group) => {
        const groupFields = FIELDS.filter((f) => f.group === group.key);
        if (groupFields.length === 0) return null;

        return (
          <div key={group.key}>
            <h4 className="form-section-title">{group.label}</h4>
            <div className="form-grid">
              {groupFields.map((field) => (
                <div className="form-group" key={field.name}>
                  <label htmlFor={`field-${field.name}`}>{field.label}</label>

                  {field.type === "number" && (
                    <input
                      id={`field-${field.name}`}
                      type="number"
                      name={field.name}
                      value={values[field.name] || ""}
                      onChange={handleChange}
                      min={field.min}
                      max={field.max}
                      step={field.step || 1}
                      required
                    />
                  )}

                  {field.type === "select" && (
                    <select
                      id={`field-${field.name}`}
                      name={field.name}
                      value={values[field.name] || ""}
                      onChange={handleChange}
                      required
                    >
                      {field.options.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  )}

                  {field.type === "yesno" && (
                    <select
                      id={`field-${field.name}`}
                      name={field.name}
                      value={values[field.name] || "0"}
                      onChange={handleChange}
                      required
                    >
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  )}
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
