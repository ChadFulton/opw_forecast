!-----------------------------------------------------------------------------
! Simulate
!-----------------------------------------------------------------------------

! - MODULES ------------------------------------------------------------------

MODULE estimation
    IMPLICIT NONE

    integer :: seed = 1234567898

CONTAINS

    SUBROUTINE get_data(lag, delta, endog, exog)
        IMPLICIT NONE

        integer, intent(in) :: lag
        integer, intent(in) :: delta
        integer, dimension(:), allocatable, intent(out) :: endog
        real(8), dimension(:,:), allocatable, intent(out) :: exog

        integer :: fst, rdst
        integer :: t=1, k
        real(8), dimension(618,58) :: raw

        ! Open the file
        OPEN(unit=10, file="recession probit data.csv", status="old",            &
            action="read", position="rewind", iostat=fst)
        ! Read the lines into an array
        DO
            READ(10, "(58f12.3)", iostat=rdst) raw(t, :)
            t = t + 1
            
            IF (rdst > 0) STOP "read error"
            IF (rdst < 0) EXIT
        END DO
        ! Close the file
        CLOSE(10)

        ALLOCATE(endog(613))
        ALLOCATE(exog(613,56))

        ! Create the endog and exog arrays
        ! TODO replace with array operations
        DO t = 1,613
            ! Endog
            endog(t) = int(raw(delta+lag+1+t, 2))

            ! Exog[:,1] = ones()
            exog(t, 1) = 1
            ! Exog[:,2] = ffr
            exog(t, 2) = raw(delta+1+t, 3)
            ! Exog[:,3] = growth_rate(sp500_return)
            exog(t, 3) = (raw(delta+1+t, 4) / raw(1+t, 4) - 1)*100
            ! Exog[:,4] = term_spread
            exog(t, 4) = raw(delta+1+t, 5) - raw(delta+1+t, 6)
            ! Exog[:,5-56] = growth_rate(agg_emp, agg_ip, 51 states)
            DO k=7,58
                exog(t, k-2) = (raw(delta+t, k) / raw(t, k) - 1)*100
            END DO
        END DO

        RETURN
    END SUBROUTINE get_data

    SUBROUTINE get_indices(gamma, idx)
        integer, dimension(:), intent(in):: gamma
        integer, dimension(:), allocatable, intent(out) :: idx
        integer, dimension(:), allocatable :: seq
        integer :: i, K, K_gamma

        K = SIZE(gamma)
        K_gamma = SUM(gamma)

        ALLOCATE(idx(K_gamma))
        ALLOCATE(seq(K))

        seq = (/(i,i=1,K)/)
        idx = PACK(seq, gamma == 1)

        DEALLOCATE(seq)

        RETURN
    END SUBROUTINE get_indices

    SUBROUTINE draw_gamma(gamma, rvs, gamma_star)
        IMPLICIT NONE

        integer, dimension(:), intent(in) :: gamma
        integer,               intent(in) :: rvs
        integer, dimension(:), intent(out) :: gamma_star

        gamma_star = gamma
        
        IF (rvs > 1) THEN
            IF (gamma_star(rvs) == 1) THEN
                gamma_star(rvs) = 0
            ELSE
                gamma_star(rvs) = 1
            ENDIF
        ENDIF

        RETURN
    END SUBROUTINE draw_gamma

    FUNCTION draw_rho(M0, y, exog, sigma2)
        IMPLICIT NONE

        real(8), dimension(:,:), intent(in) :: M0
        real(8), dimension(:),   intent(in) :: y
        real(8), dimension(:,:), intent(in) :: exog
        real(8),                 intent(in) :: sigma2
        
        real(8), dimension(:), allocatable   :: draw_rho
        real(8), dimension(:,:), allocatable :: M1
        real(8), dimension(:), allocatable   :: tmp

        integer, dimension(2) :: exog_shape
        integer :: N, K, i, info
        real(8) :: alpha = 1., beta = 0.

        exog_shape = SHAPE(exog)
        N = exog_shape(1)
        K = exog_shape(2)

        ALLOCATE(draw_rho(K))
        ALLOCATE(M1(K,K))
        ALLOCATE(tmp(K))

        draw_rho = 0.
        M1 = 0.
        tmp = 0.

        ! M1 = M0 + exog' * exog
        ! M1 is K x K
        M1 = M0(1:K,1:K) / sigma2
        CALL dgemm("T", "N", K, K, N, alpha, exog, N, exog, N, alpha, M1, K)

        ! M1 = chol(M1, "upper")
        CALL dpotrf("U", K, M1, K, info)

        ! M1 = inv(M1)
        CALL dpotri("U", K, M1, K, info)

        ! draw_rho = exog' * y;
        CALL dgemv("T", N, K, alpha, exog, N, y, 1, beta, tmp, 1)

        ! draw_rho = M1 * tmp = M1 * exog' * y
        CALL dsymv("U", K, alpha, M1, K, tmp, 1, beta, draw_rho, 1)

        ! M1 = chol(inv(M1), "upper")
        CALL dpotrf("U", K, M1, K, info)

        ! Get the random draws
        DO i=1,K
            CALL normal_01_sample(seed, tmp(i))
        END DO

        ! rho = rho + M1 * tmp;
        CALL dtrmv("U", "T", "N", K, M1, K, tmp, 1)
        draw_rho = draw_rho + tmp

        DEALLOCATE(M1)
        DEALLOCATE(tmp)

        RETURN
    END FUNCTION draw_rho

    real(8) FUNCTION ln_mvn_density(M0, sigma2, y, exog)
        IMPLICIT NONE

        real(8), dimension(:,:), intent(in) :: M0
        real(8),                 intent(in) :: sigma2
        real(8), dimension(:),   intent(in) :: y
        real(8), dimension(:,:), intent(in) :: exog

        real(8), dimension(:,:), allocatable :: Sigma
        real(8), dimension(:), allocatable   :: tmp

        integer, dimension(2) :: exog_shape
        integer :: N, K, i, info
        real(8) :: ddot, log_det = 0., alpha = 1.

        exog_shape = SHAPE(exog)
        N = exog_shape(1)
        K = exog_shape(2)

        ALLOCATE(Sigma(N,N))
        ALLOCATE(tmp(N))

        ! Sigma = M0 + sigma2 * (exog * exog');
        Sigma = M0
        CALL dgemm("N", "T", N, N, K, sigma2, exog, N, exog, N, alpha, Sigma, N)

        ! Sigma = chol(Sigma);
        CALL dpotrf("L", N, Sigma, N, info)

        ! det = det(Sigma)
        log_det = 0.
        DO i = 1,N
            log_det = log_det + log(Sigma(i,i))
        END DO

        ! tmp = y
        tmp = y

        ! Solve Sigma * tmp = tmp
        CALL dpotrs('L', N, 1, Sigma, N, tmp, N, info)

        ! Calculate the density
        ln_mvn_density = - log_det - 0.5 * ddot(N, tmp, 1, tmp, 1)

        DEALLOCATE(Sigma)
        DEALLOCATE(tmp)
        
        RETURN
    END FUNCTION ln_mvn_density

    real(8) FUNCTION lnfact(n)
        IMPLICIT NONE

        integer, intent(in) :: n
        integer :: i

        lnfact = 0.
        DO i=1,n
            lnfact = lnfact + log(real(i))
        END DO

        RETURN
    END FUNCTION lnfact

    real(8) FUNCTION ln_mn_mass(gamma)
        IMPLICIT NONE
        integer, dimension(:), intent(in) :: gamma
        integer :: K, s

        K = SIZE(gamma)
        s = SUM(gamma)

        ln_mn_mass = -(lnfact(K) - lnfact(s) - lnfact(K-s))
    
        RETURN
    END FUNCTION ln_mn_mass

    SUBROUTINE draw_y(rho, endog, exog, y)
        IMPLICIT NONE
        real(8), dimension(:),   intent(in) :: rho
        integer, dimension(:),   intent(in) :: endog
        real(8), dimension(:,:), intent(in) :: exog
        real(8), dimension(:),   intent(out) :: y

        real(8), dimension(:), allocatable :: xB

        integer, dimension(2) :: exog_shape
        integer :: i, t, N, K
        real(8) :: x, s = 1., alpha = 1., beta = 0.

        exog_shape = SHAPE(exog)
        N = exog_shape(1)
        K = exog_shape(2)

        ALLOCATE(xB(N))

        ! xB = exog * rho
        CALL dgemv("N", N, K, alpha, exog, N, rho, 1, beta, xB, 1)

        i = 0
        DO t = 1,N
            DO
                CALL normal_01_sample(seed, x)
                y(t) = xB(t) + x
                IF (endog(t) == 1 .and. y(t) > 0) THEN
                    EXIT
                ELSE IF (endog(t) == 0 .and. y(t) < 0) THEN
                    EXIT
                END IF

                i = i+1

                IF (i > 50) THEN
                    IF (endog(t) == 1) THEN
                        CALL truncated_normal_a_sample(xB(t), s, beta, seed, y(t))
                    ELSE
                        CALL truncated_normal_b_sample(xB(t), s, beta, seed, y(t))
                    END IF
                    EXIT
                END IF
            END DO
        END DO

        DEALLOCATE(xB)

        RETURN
    END SUBROUTINE draw_y

    real(8) FUNCTION calculate_accept(y, exog, M0, gamma, gamma_star, sigma2)
        IMPLICIT NONE
        real(8), dimension(:),   intent(in) :: y
        real(8), dimension(:,:), intent(in) :: exog
        real(8), dimension(:,:), intent(in) :: M0
        integer, dimension(:),   intent(in) :: gamma
        integer, dimension(:),   intent(in) :: gamma_star
        real(8),                 intent(in) :: sigma2

        integer, dimension(:), allocatable   :: idx
        real(8), dimension(:,:), allocatable :: exog_numer
        real(8), dimension(:,:), allocatable :: exog_denom

        integer, dimension(2) :: exog_shape
        integer :: N, K
        real(8) :: denom, numer

        exog_shape = SHAPE(exog)
        N = exog_shape(1)
        K = exog_shape(2)

        ALLOCATE(exog_numer(N,SUM(gamma)))
        ALLOCATE(exog_denom(N,SUM(gamma_star)))

        CALL get_indices(gamma, idx)
        exog_denom = exog(:, idx)
        DEALLOCATE(idx)

        CALL get_indices(gamma_star, idx)
        exog_numer = exog(:, idx)
        DEALLOCATE(idx)

        denom = ln_mn_mass(gamma(2:)) + ln_mvn_density(M0, sigma2, y, exog_denom)
        numer = ln_mn_mass(gamma_star(2:)) + ln_mvn_density(M0, sigma2, y, exog_numer)

        calculate_accept = exp(numer - denom)

        DEALLOCATE(exog_numer)
        DEALLOCATE(exog_denom)

        RETURN
    END FUNCTION calculate_accept

    SUBROUTINE sample(exog, endog, M0, sigma2, y, accept, gamma, rho)
        IMPLICIT NONE
        real(8), dimension(:,:), intent(in)  :: exog
        integer, dimension(:),   intent(in)  :: endog
        real(8), dimension(:,:), intent(in)  :: M0
        real(8),                 intent(in)  :: sigma2

        real(8), dimension(:),   intent(out) :: y
        logical,                 intent(out) :: accept

        integer, dimension(:),   intent(inout)  :: gamma
        real(8), dimension(:),   intent(inout)  :: rho

        integer, dimension(:), allocatable :: gamma_star
        real(8), dimension(:), allocatable :: rho_star
        integer, dimension(:), allocatable :: idx

        integer, dimension(2) :: exog_shape
        integer :: N, K, gamma_rvs
        real(8) :: r, prob_accept

        exog_shape = SHAPE(exog)
        N = exog_shape(1)
        K = exog_shape(2)

        ! Allocate temporary arrays
        ALLOCATE(gamma_star(K))
        ALLOCATE(rho_star(K))
        gamma_star = 0
        rho_star = 0.

        ! 1. Gibbs step: draw y
        CALL get_indices(gamma, idx)
        y = 0.
        CALL draw_y(rho(idx), endog, exog(:, idx), y)
        DEALLOCATE(idx)

        ! 2. Metropolis step: draw gamma and rho
        CALL RANDOM_NUMBER(r)
        gamma_rvs = CEILING(r*K)

        IF (gamma_rvs > 1) THEN
            CALL draw_gamma(gamma, gamma_rvs, gamma_star)
            prob_accept = calculate_accept(y, exog, M0, gamma, gamma_star, sigma2)
            !prob_accept = 0.5
        ELSE
            gamma_star = gamma
            prob_accept = 1
        ENDIF

        CALL RANDOM_NUMBER(r)
        accept = prob_accept >= r

        IF (accept) THEN
            gamma = gamma_star
            CALL get_indices(gamma, idx)

            rho = 0.
            rho(idx) = draw_rho(M0, y, exog(:, idx), sigma2)

            DEALLOCATE(idx)
        ENDIF

        DEALLOCATE(gamma_star)
        DEALLOCATE(rho_star)

        RETURN
    END SUBROUTINE sample

    SUBROUTINE mh(exog, endog, sigma2, ys, gammas, rhos, accepts)
        IMPLICIT NONE
        real(8), dimension(:,:), intent(in) :: exog
        integer, dimension(:),   intent(in) :: endog
        real(8),                 intent(in) :: sigma2
        real(8), dimension(:,:), intent(out) :: ys
        integer, dimension(:,:), intent(out) :: gammas
        real(8), dimension(:,:), intent(out) :: rhos
        logical, dimension(:),   intent(out) :: accepts

        integer, dimension(2) :: ys_shape
        integer :: N, iterations, t
        real(8), dimension(:,:), allocatable :: M0

        ys_shape = SHAPE(ys)
        N = ys_shape(1)
        iterations = ys_shape(2)

        ALLOCATE(M0(N, N))
        
        M0 = 0.
        DO t=1,N
            M0(t,t) = 1.
        END DO

        DO t=2,iterations
            gammas(:, t) = gammas(:, t-1)
            rhos(:, t) = rhos(:, t-1)
            CALL sample(exog, endog, M0, sigma2, ys(:,t), accepts(t), gammas(:, t), rhos(:,t))
        END DO

        DEALLOCATE(M0)

        RETURN
    END SUBROUTINE mh

END MODULE estimation

! - MAIN PROGRAM -------------------------------------------------------------

PROGRAM simulate
    USE estimation
    IMPLICIT NONE

    integer, dimension(:), allocatable   :: endog
    real(8), dimension(:,:), allocatable :: exog
    integer, dimension(2) :: exog_shape

    real(8), dimension(:,:), allocatable :: ys
    integer, dimension(:,:), allocatable :: gammas
    real(8), dimension(:,:), allocatable :: rhos
    logical, dimension(:), allocatable   :: accepts
    real(8) :: sigma2 = 10.

    integer, dimension(:), allocatable :: seq
    integer :: M, i, N, K, G0, G, iterations
    integer :: start_time, end_time
    real(8) :: elapsed, minutes, seconds

    ! Seed for uniform random draws
    CALL RANDOM_SEED(size = M)
    ALLOCATE(seq(M))
    seq = (/(i,i=1,M)/)
    CALL RANDOM_SEED(put=seq)
    DEALLOCATE(seq)

    ! Load the dataset
    CALL get_data(1, 3, endog, exog)
    exog_shape = SHAPE(exog)

    ! Parameters
    N = exog_shape(1)
    K = exog_shape(2)
    G0 = 200
    G = 200
    iterations = G0 + G + 1

    ! Allocate the data arrays
    ALLOCATE(ys(N, iterations))
    ALLOCATE(gammas(K, iterations))
    ALLOCATE(rhos(K, iterations))
    ALLOCATE(accepts(iterations))

    ! Initialize data arrays
    ys = 0.
    gammas = 0
    rhos = 0.
    accepts = .false.
    gammas(1,:) = 1

    ! Run Metropolis-Hastings
    CALL system_clock(start_time)
    CALL mh(exog, endog, sigma2, ys, gammas, rhos, accepts)
    CALL system_clock(end_time)

    ! Timing
    elapsed = real(end_time - start_time) / 1000
    minutes = int(elapsed) / 60
    seconds = MOD(elapsed, 60.)

    ! Output
    WRITE(*,*) "Runtime of ", minutes, " minutes and ", seconds, " seconds"
    WRITE(*,*) "Number of draws to convergence: ", G0
    WRITE(*,*) "Number of draws after convergence: ", G
    WRITE(*,*) "Prior VC matrix for model parameters is: ", sigma2
    WRITE(*,*) "Average Model Size: ", real(SUM(gammas(:,2:))) / real(iterations-1)

END PROGRAM simulate

subroutine init_random_seed()
    use iso_fortran_env, only: int64
    implicit none
    integer, allocatable :: seed(:)
    integer :: i, n = 4, un, istat, dt(8), pid
    integer(int64) :: t
  
    call random_seed(size = n)
    allocate(seed(n))
    ! First try if the OS provides a random number generator
    open(newunit=un, file="/dev/urandom", access="stream", &
         form="unformatted", action="read", status="old", iostat=istat)
    if (istat == 0) then
       read(un) seed
       close(un)
    else
       ! Fallback to XOR:ing the current time and pid. The PID is
       ! useful in case one launches multiple instances of the same
       ! program in parallel.
       call system_clock(t)
       if (t == 0) then
          call date_and_time(values=dt)
          t = (dt(1) - 1970) * 365_int64 * 24 * 60 * 60 * 1000 &
               + dt(2) * 31_int64 * 24 * 60 * 60 * 1000 &
               + dt(3) * 24_int64 * 60 * 60 * 1000 &
               + dt(5) * 60 * 60 * 1000 &
               + dt(6) * 60 * 1000 + dt(7) * 1000 &
               + dt(8)
       end if
       pid = getpid()
       t = ieor(t, int(pid, kind(t)))
       do i = 1, n
          seed(i) = lcg(t)
       end do
    end if
    call random_seed(put=seed)
  contains
    ! This simple PRNG might not be good enough for real work, but is
    ! sufficient for seeding a better PRNG.
    function lcg(s)
      integer :: lcg
      integer(int64) :: s
      if (s == 0) then
         s = 104729
      else
         s = mod(s, 4294967296_int64)
      end if
      s = mod(s * 279470273_int64, 4294967291_int64)
      lcg = int(mod(s, int(huge(0), int64)), kind(0))
    end function lcg
end subroutine init_random_seed