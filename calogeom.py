import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import transform
from caloplot import xy_calo_plot, encap_plot


def make_q(x, ntheta):
    return (np.hypot(x, 1.) - x)**(1. / ntheta)


def distance_sq_bw_line_and_point(p1, p2, s):
    """ Squred distance between the line drawn between p1 and p2 and the point s """
    return np.sum(np.cross(p2 - p1, p1 - s)**2) / np.sum((p2 - p1)**2)


def cry_theta_face_size(cry):
    return np.sqrt(np.min([
        distance_sq_bw_line_and_point(cry[4], cry[5], cry[6]),
        distance_sq_bw_line_and_point(cry[4], cry[5], cry[7])
    ]))


def cry_phi_face_size(cry):
    return np.sqrt(np.min([
        np.sum((cry[4] - cry[5])**2),
        np.sum((cry[6] - cry[7])**2),
    ]))


def build_crystal(x, q, i, l, nphi, barrel=True, splitlvl=1):
    dphi0 = np.pi / nphi
    cdphi0 = np.cos(dphi0 / splitlvl)
    sdphi0 = np.sin(dphi0 / splitlvl)
    tpsih1 = q**i

    psi1 = 2. * np.arctan(tpsih1)
    psi2 = 2. * np.arctan(tpsih1 * q)

    if barrel:
        th1 = np.arctan(np.tan(psi1) / cdphi0) if i else 0.5 * np.pi
    else:  # endcap
        psi1, psi2 = [0.5 * np.pi - p for p in [psi2, psi1]]  # swap!
        if splitlvl != 1:  # correction if sector is splitted
            psi1, psi2 = [np.arctan(np.tan(p) * cdphi0 / np.cos(dphi0))
                          for p in [psi2, psi1]]
        th1 = np.arctan(np.tan(psi1) / cdphi0)
    th2 = np.arctan(np.tan(psi2) / cdphi0)

    psiax = 0.5 * (psi1 + psi2)
    sth1, sth2 = np.sin(th1), np.sin(th2)
    cth1, cth2 = np.cos(th1), np.cos(th2)

    spsi1, spsi2 = np.sin(psi1), np.sin(psi2)
    cpsi1, cpsi2 = np.cos(psi1), np.cos(psi2)

    tau1 = np.array([0., spsi1, cpsi1])
    tau2 = np.array([0., spsi2, cpsi2])
    
    # tau - unit vectors, giving the directions of crystal edges
    taup1 = np.array([-sdphi0 * sth1, cdphi0 * sth1, cth1])
    taup2 = np.array([-sdphi0 * sth2, cdphi0 * sth2, cth2])
    taum1 = np.array([-taup1[0], taup1[1], taup1[2]])
    taum2 = np.array([-taup2[0], taup2[1], taup2[2]])
    tauax = np.array([0., np.sin(psiax), np.cos(psiax)])  # axis direction

    # the distance from the origin to the crystal front face
    s = x * (tauax @ tau2) / spsi2 if barrel else\
        x * (tauax @ tau1) / cpsi1
    
    tauaxtau1 = tauax @ taum1
    tauaxtau2 = tauax @ taum2
    # natural parameter along tau_p, giving the position of crystal vertex at front face
    s1f = s / tauaxtau1
    s2f = s / tauaxtau2
    # natural parameter along tau_p, giving the position of crystal vertex at back face
    s1b = (s + l) / tauaxtau1
    s2b = (s + l) / tauaxtau2

    cry = np.vstack([taum1 * s1f, taup1 * s1f, taum2 * s2f, taup2 * s2f,
                     taum1 * s1b, taup1 * s1b, taum2 * s2b, taup2 * s2b])

    if not barrel and split_level(cry) > 3:
        return build_crystal(x, q, i, l, nphi, barrel, splitlvl * 2)

    return cry, splitlvl


def split_level(cry):
    return max(1, int(cry_phi_face_size(cry) / cry_theta_face_size(cry)))


def make_crystal_surf(rho, tanf2, d):
    x = rho * tanf2
    return np.array([
        [+x, rho, -d],
        [-x, rho, -d],
        [+x, rho, +d],
        [-x, rho, +d],
    ])


def make_crystal(rho, tanf2, d, l):
    return np.vstack([
        make_crystal_surf(rho, tanf2, d),
        make_crystal_surf(rho + l, tanf2, d),
    ])


def reflectz(cry):
    rcry = cry[[1, 0, 3, 2, 5, 4, 7, 6]]
    rcry[:, 2] = -rcry[:, 2]
    return rcry


def crystal_axis(cry):
    """
        TVector3 axis_ip = (m_p[0]+m_p[1]+m_p[2]+m_p[3])*0.25;
        Plane3D sigma(m_p[4], m_p[5], m_p[6]);
        Plane3D sigma0(m_p[0], m_p[1], m_p[2]);
        Line3D l(axis_ip, axis_ip+sigma0.Normal());
        return Line3D(axis_ip, sigma.CrossPoint(l));
    """
    return np.vstack([
        cry[:4].mean(axis=0),
        cry[4:].mean(axis=0)
    ])


def vec_phi(v):
    return np.arctan2(v[1], v[0])


def delta_phi(v1, v2):
    dphi = vec_phi(v2) - vec_phi(v1)
    if dphi > np.pi:
        dphi -= 2. * np.pi
    elif dphi < -np.pi:
        dphi += 2. * np.pi
    return np.abs(dphi)


def split_endcap_crystal(cry):
    splitlvl = split_level(cry)
    if splitlvl == 1:
        return [cry]

    dphi = delta_phi(cry[5] - cry[1], cry[4] - cry[0])
    cryax = crystal_axis(cry)
    phiax = vec_phi(cryax[1] - cryax[0]) - 0.5 * np.pi - 0.5 * dphi
    focus = closest_approach_point((cry[0], cry[4]), (cry[1], cry[5]))
    lines = [(cry[2 * i], cry[2 * i + 1]) for i in range(4)]

    segment = []
    for spl in range(splitlvl):
        scry = np.empty((8, 3))
        scry[::2] = (segment[-1][1::2] if spl else cry[::2])
    
        phinorm = phiax + dphi * (spl + 1) / splitlvl
        plane = (focus, np.array([np.cos(phinorm), np.sin(phinorm), 0.]))
        scry[1::2] = np.vstack([plane_cross_line(plane, item).reshape(-1, 3)
                                for item in lines])
        segment.append(scry)
    return np.array(segment)


def plane_cross_line(plane, line):
    """ plane = (vec, norm), line = (s, t) """
    dire, norm = line[1] - line[0], plane[1]
    return line[0] + dire * ((plane[0] - line[0]) @ norm) / (dire @ norm)


def closest_approach_point(line1, line2):
    """ line1 = (s1, t1), line2 = (s2, t2) """
    dir1, dir2 = [x[1] - x[0] for x in [line1, line2]]
    dir1, dir2 = [x / np.sqrt(np.sum(x**2)) for x in [dir1, dir2]]
    assert np.sum(np.cross(dir1, dir2)**2) > 1e-8
    dirdot = dir1 @ dir2
    xi = (line1[0] - line2[0]) @ (dir2 * dirdot - dir1) / (1. - dirdot**2)
    return line1[0] + dir1 * xi


def generate_barrel_ecl(rho, z0, l, r0, nphi, ntheta):
    q = make_q(z0 / rho, ntheta)
    d_central_half = rho * (1 - q) / (1 + q) * (1. + l / rho)  # width of central ring on theta
    # recalculating with z0 -> z0-d_central_half to allow the fit exactly in z0
    q = make_q((z0 - d_central_half) / rho, ntheta)
    tanf2 = np.tan(np.pi / nphi)

    offset = np.array([r0, 0., 0.])
    crystal = make_crystal(rho, tanf2, d_central_half, l)
    rotators = [transform.Rotation.from_euler('z', iphi, degrees=False)
                for iphi in np.linspace(0, 2.*np.pi, nphi)]
    crystals = [phirot.apply(crystal + offset) for phirot in rotators]
    
    offset = np.array([r0, 0., d_central_half])
    for itheta in range(ntheta):
        cry = build_crystal(rho, q, itheta, l, nphi)[0] + offset
        for phirot in rotators:
            cry0 = phirot.apply(cry)
            crystals += [cry0, reflectz(cry0)]

    return np.array(crystals)


def generate_endcap_ecl(rho, z0, l, r0, nphi, ntheta, rhoin):
    endcapmap = {}

    dphi = np.pi / nphi
    ymax = rho * np.cos(dphi)
    q = make_q(ymax / z0, ntheta)
    dhalf = z0 * (1 - q) / (1 + q)
    q = make_q(ymax / (z0 - dhalf), ntheta)

    offset = np.array([r0, 0, dhalf])
    lendcap, rendcap = [], []
    for ith in range(1, ntheta):
        cry, splitlvl = build_crystal(z0 - dhalf, q, ith, l, nphi, barrel=False)
        cry += offset  # defocusing
        if cry[2, 1] < rhoin:
            continue

        rot0 = -dphi / splitlvl + dphi
        for spl in range(splitlvl):
            for iphi, phi in enumerate(np.linspace(0, 2.*np.pi, nphi)):
                rotangle = rot0 - 2. * dphi * spl / splitlvl
                rotator = transform.Rotation.from_euler('z', rotangle)
                cry0 = rotator.apply(cry)
                segment = split_endcap_crystal(cry0)
                for icry, segcry in enumerate(segment):
                    lsegcry = reflectz(segcry)
                    rotator = transform.Rotation.from_euler('z', phi)
                    rcry, lcry = [rotator.apply(x) for x in [segcry, lsegcry]]
                    endcapmap[rcry.tobytes()] = (iphi, spl, ith, icry, 'r')
                    endcapmap[lcry.tobytes()] = (iphi, spl, ith, icry, 'l')
                    rendcap.append(rcry)
                    lendcap.append(lcry)

    return np.array(lendcap), np.array(rendcap), endcapmap


def fill_endcap_gaps(endcap, endcapmap):
    for cry1 in endcap:
        info1 = endcapmap[cry1.tobytes()]
        for cry2 in endcap:
            pass




cfg = {
    'crystal_length': 330,
    'r_internal_bar': 1090,
    'z_calorimeter_bar': 1260,
    'z_calorimeter_end': 1290,
    'r_defocus': 20,
    'n_phi_crystals_bar': 114,
    'n_phi_sectors_end': 19,
    'n_theta_half_crystals_bar': 19,
    'd_mylar_bar': 0.2,
    'd_teflon_bar': 0.05,
    'readout_box_bar': 60.0,
    'draw_readout_box_bar': 0,
    'r_internal_end': 300.0,
}


if __name__ == '__main__':
    barrel = generate_barrel_ecl(
        cfg['r_internal_bar'],
        cfg['z_calorimeter_bar'],
        cfg['crystal_length'],
        cfg['r_defocus'],
        cfg['n_phi_crystals_bar'],
        cfg['n_theta_half_crystals_bar'],
    )

    lendcap, rendcap, endcapmap = generate_endcap_ecl(
        cfg['r_internal_bar'],
        cfg['z_calorimeter_end'],
        cfg['crystal_length'],
        cfg['r_defocus'],
        cfg['n_phi_sectors_end'],
        cfg['n_theta_half_crystals_bar'],
        cfg['r_internal_end'],
    )

    # np.save('barrel', barrel)

    # print(lendcap[50])
    # print(lendcap.shape)
    # print(rendcap.shape)
    xy_calo_plot(barrel)
    encap_plot(rendcap, False)
    # encap_plot(lendcap, True)
    plt.show()
