import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import transform
from caloplot import xy_calo_plot


def make_q(z0, rho, ntheta):
    return (np.hypot(z0 / rho, 1.) - z0 / rho)**(1./ntheta)


def distance_sq_bw_line_and_point(p1, p2, s):
    """ Squred distance between the line drawn between p1 and p2 and the point s """
    return np.sum(np.cross(p2 - p1, p1 - s)**2) / np.sum((p2 - p1)**2)


def cry_theta_face_size(cry):
    return np.sqrt(np.min(
        distance_sq_bw_line_and_point(cry[4], cry[5], cry[6]),
        distance_sq_bw_line_and_point(cry[4], cry[5], cry[7])
    ))

def cry_phi_face_size(cry):
    return np.sqrt(np.min(
        np.sum((cry[4] - cry[5])**2),
        np.sum((cry[6] - cry[7])**2),
    ))

def build_crystal(x, q, i, l, nphi, barrel=True, splitlvl=1):
    dphi0 = np.pi / nphi
    cdphi0 = np.cos(dphi0 / splitlvl)
    sdphi0 = np.sin(dphi0 / splitlvl)
    tpsih1 = q**i
    tpsih2 = tpsih1 * q

    psi1 = 2. * np.arctan(tpsih1)
    psi2 = 2. * np.arctan(tpsih2)

    if barrel:
        th1 = np.arctan(np.tan(psi1) / cdphi0) if i else 0.5 * np.pi
    else:  # endcap
        psi1 = 0.5 * np.pi - psi2  # swap!
        psi2 = 0.5 * np.pi - psi1
        if splitlvl != 1:
            # correction psi if there was a sector splitting
            psi1 = np.arctan(np.tan(psi1) * cdphi0 / np.cos(dphi0))
            psi2 = np.arctan(np.tan(psi2) * cdphi0 / np.cos(dphi0))
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
    
    # axis direction
    tauax = np.array([0., np.sin(psiax), np.cos(psiax)])

    # the distance from the origin to the crystal front face
    s = x * (tauax @ tau2) / spsi2 if barrel else\
        x * (tauax @ tau1) / spsi1
    
    tauaxtau1 = tauax @ taum1
    tauaxtau2 = tauax @ taum2
    # natural parameter along tau_p, giving the position of crystal vertex at front face
    s1 = s / tauaxtau1
    s2 = s / tauaxtau2
    # natural parameter along tau_p, giving the position of crystal vertex at back face
    s1b = (s + l) / tauaxtau1
    s2b = (s + l) / tauaxtau2

    cry = np.vstack([
        taum1 * s1, taup1 * s1, taum2 * s2, taup2 * s2,
        taum1 * s1b, taup1 * s1b, taum2 * s2b, taup2 * s2b
    ])

    if not barrel and cry_phi_face_size(cry) / cry_theta_face_size(cry) > 4.:
        return build_crystal(x, q, i, l, nphi, barrel, splitlvl * 2)

    return cry


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


def generate_barrel_ecl(rho, z0, l, r0, nphi, ntheta):
    q = make_q(z0, rho, ntheta)
    d = 2. * rho * (1 - q) / (1 + q)
    d_central_half = 0.5 * d * (1. + l / rho)  # width of central ring on theta
    # recalculating with z0 -> z0-d_central_half to allow the fit exactly in z0
    q = make_q(z0 - d_central_half, rho, ntheta)

    tanf2 = np.tan(np.pi / nphi)

    offset = np.array([r0, 0., 0.])
    phivec = np.linspace(0, 2.*np.pi, nphi)

    crystal = make_crystal(rho, tanf2, d_central_half, l)
    rotators = [transform.Rotation.from_euler('z', iphi, degrees=False)
                for iphi in phivec]
    crystals = [phirot.apply(crystal + offset) for phirot in rotators]
    
    offset = np.array([r0, 0., d_central_half])
    for itheta in range(ntheta):
        cry = build_crystal(rho, q, itheta, l, nphi) + offset
        for phirot in rotators:
            cry0 = phirot.apply(cry)
            crystals += [cry0, reflectz(cry0)]

    crystals = np.array(crystals)
    np.save('crystalls', crystals)
    xy_calo_plot(crystals)
    plt.show()


def generate_endcap_ecl(rho, z0, l, r0, nphi, ntheta, rhoin):
    dphi = np.pi / nphi
    ymax = rho * np.cos(dphi)
    q = make_q(ymax, rho, ntheta)
    d = 2. * z0 * (1 - q) / (1 + q)
    q = make_q(ymax - d, rho, ntheta)

    for ith in range(1, ntheta):
        crystall = build_crystal(z0 - 0.5 * d, q, ith, l, nphi)
        if crystall[2, 1] < rhoin:
            continue


cfg = {
    'crystal_length_bar': 330,
    'r_internal_bar': 1090,
    'z_calorimeter_bar': 1260,
    'r_defocus_bar': 20,
    'n_phi_crystals_bar': 114,
    'n_theta_half_crystals_bar': 19,
    'd_mylar_bar': 0.2,
    'd_teflon_bar': 0.05,
    'readout_box_bar': 60.0,
    'draw_readout_box_bar': 0,
}


if __name__ == '__main__':
    generate_barrel_ecl(
        cfg['r_internal_bar'],
        cfg['z_calorimeter_bar'],
        cfg['crystal_length_bar'],
        cfg['r_defocus_bar'],
        cfg['n_phi_crystals_bar'],
        cfg['n_theta_half_crystals_bar']
    )
