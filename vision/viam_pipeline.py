"""Inspect and query the vision services already configured on the Viam machine.

Run: python -m vision.viam_pipeline [--camera cam] [--detector color_detect]
No robot movement or configuration changes are made.
"""
import argparse
import asyncio
import json

from viam.services.vision import VisionClient

from connection import connect


async def get_box_midpoints(machine, detector="color_detect", camera="cam"):
    service = VisionClient.from_robot(machine, detector)
    detections = await service.get_detections_from_camera(camera, timeout=10)
    return [{"label": d.class_name, "confidence": d.confidence,
             "box": [d.x_min, d.y_min, d.x_max, d.y_max],
             "midpoint_px": [(d.x_min+d.x_max)/2, (d.y_min+d.y_max)/2]}
            for d in detections]


async def inspect_pipeline(camera="cam", detector="color_detect"):
    async with await asyncio.wait_for(connect(), 25) as machine:
        resources = [{"name": r.name, "type": r.type, "subtype": r.subtype}
                     for r in machine.resource_names]
        print(json.dumps({"resources": resources}, indent=2), flush=True)
        for resource in resources:
            if resource["subtype"] != "vision":
                continue
            name = resource["name"]
            service = VisionClient.from_robot(machine, name)
            try:
                properties = await service.get_properties(timeout=10)
                result = {"service": name,
                          "detections": properties.detections_supported,
                          "object_point_clouds": properties.object_point_clouds_supported}
                if name == detector and properties.detections_supported:
                    result["boxes"] = await get_box_midpoints(machine, detector, camera)
                if properties.object_point_clouds_supported:
                    objects = await service.get_object_point_clouds(camera, timeout=15)
                    result["objects"] = [{"reference_frame": obj.geometries.reference_frame,
                                          "centers_mm": [[g.center.x, g.center.y, g.center.z]
                                                         for g in obj.geometries.geometries]}
                                         for obj in objects]
                print(json.dumps(result, indent=2), flush=True)
            except Exception as error:
                print(json.dumps({"service": name, "error": f"{type(error).__name__}: {error}"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="cam")
    parser.add_argument("--detector", default="color_detect")
    args = parser.parse_args()
    try:
        asyncio.run(inspect_pipeline(args.camera, args.detector))
    except TimeoutError:
        parser.exit(1, "Viam connection timed out. Check machine Live status and network connectivity.\n")


if __name__ == "__main__":
    main()
