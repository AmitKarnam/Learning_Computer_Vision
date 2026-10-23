"""Compare SIFT and ORB descriptor matching strategies."""

import cv2
import numpy as np

from examples.common import load_image, make_contact_sheet, save_image, show_images


def ComputeDescriptorsSIFT(img: np.ndarray):
    return cv2.SIFT_create().detectAndCompute(img, None)


def ComputerDescriptorsORB(img: np.ndarray):
    return cv2.ORB_create().detectAndCompute(img, None)


def BruteForceSearchWithRegularMatchesCrossCheckFalse(descriptorsA, descriptorsB):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return sorted(matcher.match(descriptorsA, descriptorsB), key=lambda match: match.distance)


def BruteForceSearchWithRegularMatchesCrossCheckTrue(descriptorsA, descriptorsB):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    return matcher.match(descriptorsA, descriptorsB)


def BruteForceSearchWithSiftKNN(descriptorsA, descriptorsB):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptorsA, descriptorsB, k=2)
    return [
        [pair[0]]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]


def BruteForceSearchWithORB(descriptorsA, descriptorsB):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(descriptorsA, descriptorsB, k=2)
    return [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]


def VerifyMatchesWithRANSAC(keypointsA, keypointsB, good_matches, reprojection_threshold=5.0):
    if len(good_matches) < 4:
        return [], None, None

    src_pts = np.float32([
        keypointsA[match[0].queryIdx].pt for match in good_matches
    ]).reshape(-1, 1, 2)
    dst_pts = np.float32([
        keypointsB[match[0].trainIdx].pt for match in good_matches
    ]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        reprojection_threshold,
    )

    if homography is None or mask is None:
        return [], homography, mask

    inliers = [
        match
        for match, is_inlier in zip(good_matches, mask.ravel())
        if is_inlier
    ]
    return inliers, homography, mask


def main() -> None:
    image_a = load_image("Tour_Eiffel.jpg")
    image_b = load_image("Eiffle_Tower.jpg")
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    keypoints_a, descriptors_a = ComputeDescriptorsSIFT(gray_a)
    keypoints_b, descriptors_b = ComputeDescriptorsSIFT(gray_b)
    if descriptors_a is None or descriptors_b is None:
        raise ValueError("SIFT did not find descriptors in both input images.")

    good_matches = BruteForceSearchWithSiftKNN(descriptors_a, descriptors_b)
    inliers, _, mask = VerifyMatchesWithRANSAC(keypoints_a, keypoints_b, good_matches)
    print(f"SIFT ratio-test matches: {len(good_matches)}")
    print(f"RANSAC inliers: {len(inliers)}")
    if good_matches and mask is not None:
        print(f"Inlier ratio: {len(inliers) / len(good_matches):.2%}")

    sift_matches = BruteForceSearchWithRegularMatchesCrossCheckFalse(
        descriptors_a,
        descriptors_b,
    )
    cross_checked_matches = BruteForceSearchWithRegularMatchesCrossCheckTrue(
        descriptors_a,
        descriptors_b,
    )
    orb_keypoints_a, orb_descriptors_a = ComputerDescriptorsORB(gray_a)
    orb_keypoints_b, orb_descriptors_b = ComputerDescriptorsORB(gray_b)
    if orb_descriptors_a is None or orb_descriptors_b is None:
        raise ValueError("ORB did not find descriptors in both input images.")
    orb_matches = BruteForceSearchWithORB(orb_descriptors_a, orb_descriptors_b)

    match_flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    results = [
        (
            "SIFT: brute force",
            cv2.drawMatches(image_a, keypoints_a, image_b, keypoints_b, sift_matches[:100], None, flags=match_flags),
        ),
        (
            "SIFT: cross-check",
            cv2.drawMatches(image_a, keypoints_a, image_b, keypoints_b, cross_checked_matches[:100], None, flags=match_flags),
        ),
        (
            "SIFT: ratio test",
            cv2.drawMatchesKnn(image_a, keypoints_a, image_b, keypoints_b, good_matches, None, flags=match_flags),
        ),
        (
            "SIFT: RANSAC inliers",
            cv2.drawMatchesKnn(image_a, keypoints_a, image_b, keypoints_b, inliers, None, flags=match_flags),
        ),
        (
            "ORB: ratio test",
            cv2.drawMatches(image_a, orb_keypoints_a, image_b, orb_keypoints_b, orb_matches[:100], None, flags=match_flags),
        ),
    ]
    result = make_contact_sheet(results, columns=1, tile_size=(1200, 300))
    save_image("features_and_matching/descriptor_matching/matching_methods.png", result)
    show_images([("Descriptor matching", result)])


if __name__ == "__main__":
    main()
