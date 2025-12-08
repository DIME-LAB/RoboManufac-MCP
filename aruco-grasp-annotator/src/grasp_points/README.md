# Grasp Points Pipeline

## Step 1: CAD to Grasp Points Detection

Generate grasp points from CAD models using ArUco markers:

```bash
cd src/grasp_points
uv run python cad_to_grasp_pipeline.py --object <object_name> --marker-id <marker_id>
```

Example:
```bash
uv run python cad_to_grasp_pipeline.py --object fork_orange_scaled70 --marker-id 5
```

## Step 2: Annotate to All Markers

Transform grasp points to be relative to all ArUco markers on the object:

```bash
cd src/grasp_points
uv run python annotate_grasp_to_all_markers.py --object <object_name> --source-marker-id <marker_id>
```

Example:
```bash
uv run python annotate_grasp_to_all_markers.py --object fork_orange_scaled70 --source-marker-id 5
```

## Output

Generated files are saved to `data/grasp/<object_name>_grasp_points_all_markers.json` and can be used in:
- Assembly App (web visualization)
- ArUco Localizer (camera overlay)